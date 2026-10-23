"""Per-turn rules the agent loop enforces in code rather than in the prompt."""
import asyncio
import json
import logging
import re

from msight_agent.agent.routing import WRITE_TOOLS, format_reply

MAX_AGENTIC_ITERATIONS = 8  # tools are coarse, but an action plus a few verification calls can use 5+

# Text-only tools, excluded on grounded iterations so they can't narrate instead of act.
_GROUNDING_EXEMPT_TOOLS = frozenset({"send_reply"})

# Consent gate: after a diagnosis, write tools stay off unless the user asked for a change.
CHANGE_REQUEST_RE = re.compile(
    r"\b(fix|repair|restart|restore|resolve|recover|relaunch|reset|start|stop|add|"
    r"remove|delete|re-?add|bring (it )?back|go ahead|do it|yes|yeah|yep|sure|ok|okay)\b",
    re.IGNORECASE,
)
WRITES_BLOCKED_NOTE = (
    "WRITES_DISABLED: you ran a diagnosis, but the user described a symptom without "
    "asking for a change. Report the root cause and its evidence, offer the fix, and "
    "ask before changing anything -- the pipeline-changing tools are unavailable this turn."
)

# Measure-first gate: "is it working?" can't be answered from process state, and a
# hint in the status result alone stopped being followed once status grew richer.
HEALTH_QUESTION_RE = re.compile(
    r"\b(work|works|working|healthy|health|alive|broken|"
    r"is (it|the pipeline) (actually |really )?running|"
    r"running (ok|okay|fine|properly|smoothly|correctly))\b",
    re.IGNORECASE,
)


def without_writes(tool_list: list) -> list:
    return [t for t in tool_list if t["function"]["name"] not in WRITE_TOOLS]


def measure_first(tool_list: list) -> list:
    """Offer only diagnose on a health question's first step. Removing just the
    status tool didn't work -- the model fell back to logs and still never measured."""
    diagnose = [t for t in tool_list if t["function"]["name"] == "diagnose_msight_pipeline"]
    return diagnose or tool_list


def filter_tools_for_state(all_tools: list, state) -> list:
    """Tools valid for the current workflow step; fails open on None state or errors."""
    if state is None:
        return all_tools
    try:
        valid_names = state.valid_tool_names()
    except Exception:
        logging.warning("[FILTER] valid_tool_names() raised — returning full tool list")
        return all_tools
    if valid_names is None:
        return all_tools
    return [t for t in all_tools if t["function"]["name"] in valid_names]


def iteration_tool_choice(iteration: int, force_ground_next: bool, allow_send_reply_first: bool,
                          tools: list) -> tuple[str, bool, list]:
    """(tool_choice, grounded, tools) for one loop iteration. A grounded iteration must
    call a real tool; send_reply is withheld except on a fresh session's first step."""
    grounded = iteration == 0 or force_ground_next
    if not grounded or (iteration == 0 and allow_send_reply_first):
        return ("required" if grounded else "auto"), grounded, tools
    return "required", True, [t for t in tools if t["function"]["name"] not in _GROUNDING_EXEMPT_TOOLS]


def next_force_ground(tool_results: list) -> tuple[bool, str]:
    """Whether the next iteration must be grounded, and why."""
    if len(tool_results) == 1 and tool_results[0]["name"] == "send_intro":
        return True, "send_intro without its promised follow-up call"
    failed = [r["name"] for r in tool_results if r.get("failed")]
    if failed:
        return True, f"a real tool failure ({failed})"
    return False, ""


async def chat_with_retry(llm_client, messages, **kwargs):
    """Retry once on connection/timeout errors (matched by class name across SDKs)."""
    try:
        return await llm_client.chat(messages, **kwargs)
    except Exception as e:
        name = type(e).__name__
        if "Connection" not in name and "Timeout" not in name:
            raise
        logging.warning(f"[STREAM] LLM call hit {name}: {e} -- retrying once")
        await asyncio.sleep(1)
        return await llm_client.chat(messages, **kwargs)


def llm_error_message(exc: Exception) -> str:
    if "timeout" in str(exc).lower():
        return "The request timed out reaching the AI service. Please try again."
    return "Something went wrong connecting to the AI service. Please try again."


def solo_send_reply_text(tool_call, write_succeeded: bool) -> str:
    """Reply text for a send_reply call; a successful write this turn drops the source tag."""
    try:
        args = json.loads(tool_call.function.arguments)
    except (TypeError, ValueError):
        return "Something went wrong. Please try again."
    return format_reply(args.get("message", ""), None if write_succeeded else args.get("source"))


def exhaustion_fallback(all_tool_results: list) -> str:
    """Last-resort reply; never claims failure if a write tool succeeded this turn."""
    if any(r["name"] in WRITE_TOOLS and not r.get("failed") for r in all_tool_results):
        return ("I made real progress on this before running out of steps to summarize it "
                "clearly — please check the dashboard, or ask me for the current status.")
    return "I wasn't able to complete this step. Please try again."
