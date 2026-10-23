"""One user turn of the agent: call the LLM, run its tool calls, repeat until it replies."""
import logging
from typing import Awaitable, Callable, Optional

from msight_agent.agent import guards
from msight_agent.agent.prompts import build_system_prompt
from msight_agent.agent.dispatch import ChatPipeline
from msight_agent.agent.tool_schema import tools as ALL_TOOLS
from msight_agent.agent.state import WORKFLOW_SPECS

Emit = Callable[[str, dict], Awaitable[None]]

_DATASET_REMINDER = (
    "Reminder: the user has selected a workflow but has not yet confirmed a dataset. "
    "Show the user the dataset list returned by list_datasets above. Do NOT list datasets from memory."
)
_BUDGET_EXHAUSTED_NOTE = (
    "ITERATION_BUDGET_EXHAUSTED: You've reached the step limit for this turn. Call send_reply "
    "now, in plain language, describing only what the tool results already shown above actually "
    "confirmed -- do not claim anything succeeded or failed that isn't already shown there."
)


def _assistant_tool_call_message(assistant_message) -> dict:
    return {
        "role": "assistant",
        "content": assistant_message.content or "",
        "tool_calls": [
            {"id": c.id, "type": "function",
             "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in assistant_message.tool_calls
        ],
    }


class AgentTurn:
    """`emit(event, data)` receives status/log/progress events and exactly one
    terminal "reply" or "error"."""

    def __init__(self, *, message: str, messages: list, tools: list, llm_client, mcp_client,
                 pipeline: Optional[ChatPipeline], emit: Emit):
        self.messages = messages
        self.tools = tools
        self.llm = llm_client
        self.mcp_client = mcp_client
        self.emit = emit
        self.all_tool_results: list[dict] = []
        # A pipeline built while picking the default workflow means a fresh session:
        # the model may then ask STEP 1's question instead of guessing.
        self.fresh_session = pipeline is not None
        self.pipeline = pipeline
        self.change_requested = bool(guards.CHANGE_REQUEST_RE.search(message))
        self.health_question = bool(guards.HEALTH_QUESTION_RE.search(message))
        self.writes_blocked = False

    @property
    def _write_succeeded(self) -> bool:
        return bool(self.pipeline and self.pipeline.write_succeeded_this_turn)

    async def _reply(self, text: str) -> None:
        await self.emit("reply", {"message": text})

    async def run(self) -> None:
        force_ground_next = False
        for iteration in range(guards.MAX_AGENTIC_ITERATIONS):
            tool_choice, grounded, iteration_tools = guards.iteration_tool_choice(
                iteration, force_ground_next, self.fresh_session, self.tools)
            if iteration == 0 and self.health_question:
                iteration_tools = guards.measure_first(iteration_tools)

            try:
                assistant_message = await guards.chat_with_retry(
                    self.llm, self.messages, tools=iteration_tools, tool_choice=tool_choice)
            except Exception as e:
                logging.warning(f"[STREAM] LLM call failed on iter={iteration}: {type(e).__name__}: {e}")
                await self.emit("error", {"message": guards.llm_error_message(e)})
                return

            tool_calls = getattr(assistant_message, "tool_calls", None)
            if not tool_calls:
                if grounded:
                    # Provider ignored tool_choice="required"; retry rather than accept.
                    logging.warning(f"[STREAM DECISION] iter={iteration} → provider ignored "
                                    "tool_choice='required' -- retrying grounded")
                    continue
                logging.warning(f"[STREAM DECISION] iter={iteration} → end_turn (no tool call)")
                await self._reply(assistant_message.content or "")
                return

            logging.warning(f"[STREAM DECISION] iter={iteration} tool_choice={tool_choice!r} → "
                            f"{[f'{c.function.name}({c.function.arguments[:60]})' for c in tool_calls]}")
            if len(tool_calls) == 1 and tool_calls[0].function.name == "send_reply":
                await self._reply(guards.solo_send_reply_text(tool_calls[0], self._write_succeeded))
                return

            early_reply, force_ground_next = await self._run_tools(assistant_message, iteration)
            if early_reply is not None:
                await self._reply(early_reply)
                return

        await self._wrap_up()

    async def _run_tools(self, assistant_message, iteration: int) -> tuple[Optional[str], bool]:
        """Run the tool calls, then refresh tools/prompt for the next iteration.
        Returns (early reply, whether the next iteration must be grounded)."""
        self.messages.append(_assistant_tool_call_message(assistant_message))
        if self.pipeline is None:
            self.pipeline = ChatPipeline(mcp_client=self.mcp_client, llm=self.llm)
        tool_results, early_reply = await self.pipeline.run(
            assistant_message.tool_calls, self.messages, progress_cb=self.emit)
        self.all_tool_results.extend(tool_results)
        if early_reply is not None:
            return early_reply, False

        force_ground, reason = guards.next_force_ground(tool_results)
        if force_ground:
            logging.warning(f"[STREAM] iter={iteration} had {reason} -- forcing next iteration grounded")

        state = self.pipeline.state
        self.tools = guards.filter_tools_for_state(ALL_TOOLS, state)
        self._apply_consent_gate()
        self.messages[0]["content"] = build_system_prompt(state)  # in case the workflow changed

        spec = WORKFLOW_SPECS.get(state.workflow_name)
        if (spec is not None and spec.requires_dataset and not state.dataset_confirmed
                and "list_datasets" not in (r["name"] for r in tool_results)):
            self.messages.append({"role": "system", "content": _DATASET_REMINDER})
        return None, force_ground

    def _apply_consent_gate(self) -> None:
        """After a diagnosis, pipeline-changing tools stay off unless the user asked for a change."""
        if self.change_requested or not any(
                r["name"] == "diagnose_msight_pipeline" for r in self.all_tool_results):
            return
        self.tools = guards.without_writes(self.tools)
        if not self.writes_blocked:
            self.messages.append({"role": "system", "content": guards.WRITES_BLOCKED_NOTE})
            self.writes_blocked = True
            logging.warning("[STREAM] Diagnosis without a change request -- write tools off for this turn")

    async def _wrap_up(self) -> None:
        """Out of iterations: one send_reply-only call so the model summarizes what actually happened."""
        logging.warning(f"[STREAM] Exceeded {guards.MAX_AGENTIC_ITERATIONS} agentic iterations")
        send_reply_only = [t for t in self.tools if t["function"]["name"] == "send_reply"]
        if send_reply_only:
            self.messages.append({"role": "system", "content": _BUDGET_EXHAUSTED_NOTE})
            try:
                wrap_up = await self.llm.chat(self.messages, tools=send_reply_only, tool_choice="required")
                if wrap_up.tool_calls and wrap_up.tool_calls[0].function.name == "send_reply":
                    await self._reply(guards.solo_send_reply_text(wrap_up.tool_calls[0], self._write_succeeded))
                    return
            except Exception as e:
                logging.warning(f"[STREAM] Forced wrap-up call failed: {e}")
        await self._reply(guards.exhaustion_fallback(self.all_tool_results))
