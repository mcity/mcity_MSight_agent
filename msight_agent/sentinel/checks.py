"""Code-side checks around an investigation, kept free of claude_agent_sdk imports.

check_report compares Claude's report with facts the watcher measured itself, and
flags destructive advice. Logs are untrusted text that goes into the prompt, so a
log line could push a wrong root cause or a dangerous "fix"; these checks don't
depend on the model to notice. A flagged report is still shown to a person, but
autofix_decision never acts on it.

secret_env_names lists what to drop from the sentinel's environment so the Claude
subprocess (which inherits it) only gets the credentials it needs.
"""
import re
from typing import Iterable, Optional

# The categories the watcher's own evidence allows, by incident kind. An OOM's root cause
# can be the memory itself, a limit set wrong, or someone applying that limit by hand;
# the memory-mention check below catches a report steered away from memory entirely.
EXPECTED_CATEGORY = {
    "CONTAINER_OOM": {"resource_exhaustion", "misconfiguration", "external_action"},
    "CONTAINER_KILLED": {"external_action"},
}

DESTRUCTIVE_RE = re.compile(
    r"\bpurge\b|\bapt(-get)?\s+(remove|autoremove)\b|\brm\s+-[a-z]*r[a-z]*f|\brm\s+-[a-z]*f[a-z]*r|\bmkfs\b|"
    r"\bdd\s+if=|\bdocker\s+(system|volume|image)\s+prune\b|\bdocker\s+rmi?\b|\breboot\b|"
    r"\bshutdown\s+(now|-[hrP])|\bsudo\s+shutdown\b|"
    r"\bsystemctl\s+(stop|disable|mask)\b|\bchmod\s+(-R\s+)?777\b|\bdisable\s+secure\s*boot\b|"
    r"--privileged|\bkill\s+-9\s+-1\b|\bdrop\s+(table|database)\b|\bflushall\b",
    re.IGNORECASE,
)

_SECRET_NAME_RE = re.compile(r"KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL", re.IGNORECASE)
_KEEP_PREFIXES = ("ANTHROPIC_", "CLAUDE_")   # non-secret settings for the Claude CLI
_KEEP_SECRETS = {"ANTHROPIC_API_KEY"}         # the one credential the investigator needs


def check_report(incident: dict, report: dict) -> list[str]:
    """Warnings where the report disagrees with measured facts or advises something destructive."""
    warnings = []
    state = (incident.get("evidence") or {}).get("container_state") or {}
    category = report.get("category")

    expected = EXPECTED_CATEGORY.get(incident.get("kind"))
    if expected and category not in expected:
        warnings.append(f"The watcher classified this as {incident['kind']} from Docker's own events, "
                        f"but the report's category is '{category}' (expected {' or '.join(sorted(expected))}).")
    if state.get("OOMKilled") and not re.search(r"\b(oom|memory)\b", report.get("root_cause", ""), re.IGNORECASE):
        warnings.append("Docker reports OOMKilled=true, but the root cause doesn't mention memory or OOM.")
    if state and state.get("Running") is False and _claims_running(report.get("summary", ""), incident.get("node", "")):
        warnings.append("The report suggests the node is running or recovered, but Docker shows it is not running.")

    node = incident.get("node", "")
    mentioned = " ".join([report.get("root_cause", ""), report.get("summary", ""), *report.get("affected", [])])
    if incident.get("kind", "").startswith("CONTAINER_") and node and node not in mentioned:
        warnings.append(f"The incident is about '{node}', but the report never names it.")

    if report.get("confidence") == "high" and (category == "unknown" or not report.get("evidence")):
        warnings.append("Confidence is 'high' with no evidence or an unknown category.")

    for field in ("suggested_fix", "prevention"):
        text = report.get(field, "")
        for command, negated in sorted({(m.group(0), _negated(text, m.start())) for m in DESTRUCTIVE_RE.finditer(text)}):
            if negated:
                warnings.append(f"{field} mentions '{command}' only as something NOT to do -- check the wording.")
            else:
                warnings.append(f"Destructive advice in {field}: '{command}' -- needs human review before anyone runs it.")
    return warnings


# "It is running again" / "the node recovered" -- now, not "it was running normally when it died".
_RUNNING_NOW_RE = re.compile(r"\b(is|are) (still |now |back )?(up and )?running\b|\brunning again\b"
                             r"|\bback up\b|\brecovered\b", re.IGNORECASE)
_NODE_NAME_RE = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b")


def _claims_running(summary: str, node_field: str) -> bool:
    """Whether the summary says the incident's node is running now. Sentences that only
    name other nodes ("video_source is running fine") don't count."""
    nodes = set(node_field.split("+"))
    for sentence in re.split(r"(?<=[.!?;])\s+", summary):
        match = _RUNNING_NOW_RE.search(sentence)
        if not match or re.search(r"\b(not|no longer|never)\b|n't\b", sentence[:match.end()], re.IGNORECASE):
            continue
        named = set(_NODE_NAME_RE.findall(sentence))
        if named and not named & nodes:
            continue
        return True
    return False


_NEGATION_RE = re.compile(r"\b(do not|don't|never|avoid|must not|should not|shouldn't)\b", re.IGNORECASE)


def _negated(text: str, pos: int) -> bool:
    """Whether the sentence leading up to `pos` tells the reader NOT to do it."""
    sentence_start = max(text.rfind(".", 0, pos), text.rfind("\n", 0, pos), text.rfind(";", 0, pos)) + 1
    return bool(_NEGATION_RE.search(text[sentence_start:pos]))


def secret_env_names(environ: dict, dotenv_keys: Iterable[str]) -> list[str]:
    """Variables to remove: every secret-looking name except ANTHROPIC_API_KEY, and every
    other .env entry except the Anthropic/Claude CLI settings."""
    dotenv_keys = set(dotenv_keys)
    remove = []
    for name in environ:
        if name in _KEEP_SECRETS:
            continue
        if _SECRET_NAME_RE.search(name) or (name in dotenv_keys and not name.startswith(_KEEP_PREFIXES)):
            remove.append(name)
    return sorted(remove)


# --- whether the autonomous fixer may act ------------------------------------------------

FLAP_WINDOW_SECONDS = 30 * 60
MAX_FIXES_PER_WINDOW = 2


def autofix_decision(incident: dict, report: Optional[dict], warnings: list[str],
                     past_fixes: Iterable[float], now: float,
                     max_fixes: int = MAX_FIXES_PER_WINDOW) -> tuple[bool, str]:
    """(fix it?, why not) for one diagnosed incident. Escalating is the default whenever
    the diagnosis is missing, doubtful, says a person is needed, or the node keeps failing."""
    if not report:
        return False, "There is no diagnosis to act on."
    if report.get("requires_human"):
        return False, report.get("human_action") or "The diagnosis says a person needs to handle this."
    if warnings:
        return False, "The diagnosis failed the consistency checks, so it isn't safe to act on automatically."
    if report.get("confidence") == "low":
        return False, "The diagnosis has low confidence."
    if incident.get("kind") == "SERVICE_DOWN":
        return False, "Agent services and Redis can't be restarted from the pipeline tools."
    recent = [t for t in past_fixes if now - t < FLAP_WINDOW_SECONDS]
    if len(recent) >= max_fixes:
        return False, (f"'{incident.get('node')}' was already fixed automatically {len(recent)} times in the last "
                       f"{FLAP_WINDOW_SECONDS // 60} minutes; it keeps failing, so a person should look.")
    return True, ""
