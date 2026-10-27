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


# Destructive tools run only when the user asked for that action: in their current message,
# or by agreeing to a reply that asked about it. Never as the model's own recovery step.
_STOP = r"stop|end|shut\s*down|kill|halt|terminate|tear\s*down|turn\s*off|finish"
DESTRUCTIVE_INTENT: dict[str, tuple[str, str]] = {
    # tool: (words that ask for it, words a confirmation question about it contains)
    "stop_msight_pipeline": (_STOP + r"|restart|replace|instead|switch|clean\s*up", r"stop|shut|replace"),
    "remove_msight_node": (r"remove|delete|drop|get\s*rid\s*of|clean\s*up|take\s*(it\s*)?(out|down)|"
                           r"tear\s*down|kill|stop", r"remove|delete|stop"),
    "stop_msight_recording": (_STOP, r"stop"),
    "stop_msight_archiving": (_STOP, r"stop"),
    # Also the designed way to leave a workflow ("that's all", "I'm finished"): it clears
    # only the chat session, never the pipeline or its nodes.
    "reset_workflow_state": (r"reset|start\s*over|from\s*scratch|clear\s*(everything|it\s*all)|wipe|"
                             r"new\s*session|discard|that'?s\s*all|i'?m\s*(done|finished)|exit|"
                             r"leave\s*(the|this)\s*workflow", r"reset|start\s*over|discard|exit"),
}
# Only a negation right before the verb ("don't stop", "no need to remove it"): "it's not
# working, stop it" is still a request.
_NEGATED = r"\b(don'?t|do\s*not|never|no\s*need\s*to)\s+(\w+\s+){0,2}?(%s)"
_AFFIRMATIVE_RE = re.compile(r"^\s*(yes|yeah|yep|sure|ok|okay|go\s*ahead|do\s*it|please\s*do|confirm|"
                             r"correct|that'?s\s*right)\b", re.IGNORECASE)


def _target(fn_name: str, fn_args: dict) -> str:
    return fn_args.get("name", "") if fn_name == "remove_msight_node" else ""


def destructive_allowed(fn_name: str, fn_args: dict, user_message: str, last_reply: str) -> bool:
    """Whether the user asked for this destructive call (see DESTRUCTIVE_INTENT)."""
    if fn_name not in DESTRUCTIVE_INTENT:
        return True
    asks, question = DESTRUCTIVE_INTENT[fn_name]
    message = user_message.lower()
    if re.search(rf"\b({asks})", message) and not re.search(_NEGATED % asks, message):
        return True
    # "yes" counts only for a reply that asked about this action -- and this target, if any.
    target = _target(fn_name, fn_args).lower()
    reply = last_reply.lower()
    return bool(_AFFIRMATIVE_RE.search(message) and re.search(rf"\b({question})", reply)
                and (not target or target in reply))


def confirmation_question(fn_name: str, fn_args: dict) -> str:
    return {
        "stop_msight_pipeline": "Should I stop the pipeline?",
        "remove_msight_node": f"Should I remove {_target(fn_name, fn_args) or 'that node'}?",
        "stop_msight_recording": "Should I stop the recording?",
        "stop_msight_archiving": "Should I stop archiving?",
        "reset_workflow_state": "Should I reset the session and discard all progress?",
    }[fn_name]


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
