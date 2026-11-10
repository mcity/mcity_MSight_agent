"""What the sentinel's Claude sessions may touch. The investigator: read-only tools,
no secrets. The autonomous fixer: the same, plus restarts scoped to its incident.

Kept free of claude_agent_sdk imports so the agent's test suite can check it.
The SDK sessions also run with permission_mode="dontAsk" and only these tools
enabled; the policies here are the PreToolUse hook's second line of defence.
"""
import re
from typing import Iterable, Optional

MCP_SERVER_NAME = "msight"

# Every one of these only reads state (see msight_agent/msight/tools/).
READ_ONLY_MCP_TOOLS = (
    "diagnose_msight_pipeline",
    "get_msight_status",
    "get_msight_logs",
    "inspect_host",
    "list_msight_node_types",
    "get_msight_reference",
    "get_msight_record_archive_status",
    "check_msight_calibration_status",
    "get_recent_incidents",              # history: is this fault recurring?
)
BUILTIN_TOOLS = ("Read", "Grep", "Glob")
# How the CLI hands back the output_format report; it only returns data.
REPORT_TOOL = "StructuredOutput"
ALLOWED_TOOLS = (BUILTIN_TOOLS + (REPORT_TOOL,)
                 + tuple(f"mcp__{MCP_SERVER_NAME}__{t}" for t in READ_ONLY_MCP_TOOLS))

# .env holds the API keys; deny anything that looks like a credential file.
_SECRET_PATH_RE = re.compile(
    r"(^|/)\.env([^A-Za-z0-9_]|$)|[.,{]env([},]|$)|\.pem$|\.key$"
    r"|(^|/)id_(rsa|ed25519|ecdsa)|(^|/)\.aws(/|$)|credentials|secrets?(/|$|\.)",
    re.IGNORECASE,
)
# Grep's "pattern" is the search regex, not a path, so it isn't checked.
_PATH_ARGS = {"Read": ("file_path",), "Grep": ("path", "glob"), "Glob": ("pattern", "path")}


def check_tool_call(tool_name: str, tool_input: dict) -> Optional[str]:
    """None if the call is allowed, else the reason it is denied."""
    if tool_name not in ALLOWED_TOOLS:
        return (f"'{tool_name}' is not available: this investigation is read-only. "
                "Use the diagnose/status/logs/inspect_host tools and Read/Grep/Glob.")
    for key in _PATH_ARGS.get(tool_name, ()):
        value = tool_input.get(key)
        if isinstance(value, str) and _SECRET_PATH_RE.search(value):
            return f"Reading '{value}' is not allowed: it may contain credentials."
    return None


# --- the autonomous fixer (--autonomy fix) ---------------------------------------------
# It may only bring back what the incident is about, with its normal settings: never stop or
# remove anything, never choose a source or archive destination (a steered fixer could redirect
# footage). Not start_msight_pipeline: recomposing stops a running recording and added nodes.

FIX_TOOLS = ("restart_msight_node", "start_msight_recording")
FIX_ALLOWED_TOOLS = ALLOWED_TOOLS + tuple(f"mcp__{MCP_SERVER_NAME}__{t}" for t in FIX_TOOLS)
MAX_FIX_ACTIONS = 2   # per node of the incident
RECORDING_NODES = frozenset({"frame_annotator", "video_aggregator", "local_dumper"})
_SOURCE_ARGS = ("video_input", "rtsp_url", "sensor_name", "build")


class FixPolicy:
    """The fixer's PreToolUse rule for one incident: read-only tools as for the
    investigator, plus scoped restarts. Counts actions so it can enforce the budget."""

    def __init__(self, incident_nodes: Iterable[str]):
        self.incident_nodes = frozenset(incident_nodes)
        self.budget = MAX_FIX_ACTIONS * max(1, len(self.incident_nodes))
        self.actions: list[str] = []

    def __call__(self, tool_name: str, tool_input: dict) -> Optional[str]:
        name = tool_name.removeprefix(f"mcp__{MCP_SERVER_NAME}__")
        if name not in FIX_TOOLS:
            return check_tool_call(tool_name, tool_input)
        if len(self.actions) >= self.budget:
            return (f"Only {self.budget} fix actions are allowed for this incident. "
                    "Stop and report it as escalated if the pipeline isn't fixed.")
        if name == "restart_msight_node" and tool_input.get("name") not in self.incident_nodes:
            return (f"You may only restart the node(s) this incident is about: {sorted(self.incident_nodes)}. "
                    "Report other problems instead of acting on them.")
        if any(tool_input.get(arg) for arg in _SOURCE_ARGS):
            return "Use the current settings only; changing the source or sensor needs a human."
        if name == "start_msight_recording" and not self.incident_nodes & RECORDING_NODES:
            return "Recording can only be restarted when this incident is about a recording node."
        self.actions.append(f"{name}({', '.join(f'{k}={v}' for k, v in tool_input.items())})")
        return None


def incident_nodes(node_field: str) -> list[str]:
    """The nodes an incident is about: its node field may join several with '+'."""
    return [n for n in node_field.split("+") if n]
