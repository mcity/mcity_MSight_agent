"""One Claude Agent SDK session against the MSight MCP server, shared by the investigator
(read-only) and the fixer (scoped restarts). Each streams its steps to the console and to
the live feed, and every tool call passes the caller's policy in a PreToolUse hook."""
import json
import time
from typing import Callable, Optional

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from msight_agent.sentinel.policy import MCP_SERVER_NAME, REPORT_TOOL
from msight_agent.sentinel.stream import friendly_step
from msight_agent.shared.paths import REPO_ROOT


def _short(value, limit: int = 110) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _describe_call(block: ToolUseBlock) -> str:
    name = block.name.removeprefix(f"mcp__{MCP_SERVER_NAME}__")
    args = ", ".join(f"{k}={_short(v, 70)}" for k, v in block.input.items())
    return f"{name}({args})"


async def run_session(prompt: str, *, system_prompt: str, policy: Callable[[str, dict], Optional[str]],
                      tools: list[str], allowed_tools: list[str], output_schema: dict, add_dirs: list[str],
                      say: Callable[[str], None], progress: Optional[Callable[[str], None]], model: str,
                      mcp_url: str, max_turns: int, max_budget_usd: float) -> dict:
    """One Claude Agent SDK session against the MSight MCP server. Every tool call goes
    through `policy` (None = allowed, else the reason it's denied) in a PreToolUse hook,
    on top of permission_mode="dontAsk" and the allowed-tool list."""
    denied: list[str] = []

    async def guard(hook_input, tool_use_id, context):
        reason = policy(hook_input["tool_name"], hook_input.get("tool_input") or {})
        if reason is None:
            return {}
        denied.append(f"{hook_input['tool_name']}: {reason}")
        say(f"   [denied] {hook_input['tool_name']}: {reason}")
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                       "permissionDecisionReason": reason}}

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        model=model,
        tools=tools,
        allowed_tools=allowed_tools,
        permission_mode="dontAsk",
        mcp_servers={MCP_SERVER_NAME: {"type": "sse", "url": mcp_url}},
        strict_mcp_config=True,
        setting_sources=[],   # ignore ~/.claude and project settings: same behaviour everywhere
        cwd=str(REPO_ROOT),
        add_dirs=add_dirs,
        hooks={"PreToolUse": [HookMatcher(hooks=[guard])]},
        output_format={"type": "json_schema", "schema": output_schema},
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
    )

    record: dict = {"model": model, "tool_calls": [], "denied": denied, "report": None}
    started = time.monotonic()
    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            tool_ids: set[str] = set()
            async for message in client.receive_response():
                _record_message(message, record, tool_ids, say, progress)
    except Exception as e:  # noqa: BLE001 -- the incident is still logged without a diagnosis
        record["error"] = f"{type(e).__name__}: {e}"
    record["duration_s"] = time.monotonic() - started
    return record


def _record_message(message, record: dict, tool_ids: set[str], say: Callable[[str], None],
                    progress: Optional[Callable[[str], None]] = None) -> None:
    """Stream one SDK message to the console and fold it into the record."""
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock) and block.text.strip():
                say(f"   claude: {_short(block.text, 300)}")
            elif isinstance(block, ToolUseBlock) and block.name != REPORT_TOOL:
                call = _describe_call(block)
                tool_ids.add(block.id)
                record["tool_calls"].append(call)
                say(f"   -> {call}")
                if progress:
                    progress(friendly_step(block.name, block.input))
    elif isinstance(message, UserMessage) and isinstance(message.content, list):
        for block in message.content:
            if isinstance(block, ToolResultBlock) and block.tool_use_id in tool_ids:
                content = block.content
                if isinstance(content, list):
                    content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                say(f"   {'!!' if block.is_error else '<-'} {_short(content or '', 140)}")
    elif isinstance(message, ResultMessage):
        record.update(turns=message.num_turns, cost_usd=message.total_cost_usd,
                      stop=message.subtype, text=message.result)
        if isinstance(message.structured_output, dict):
            record["report"] = message.structured_output
        if message.is_error:
            record["error"] = f"{message.subtype}: {message.errors or message.result}"
