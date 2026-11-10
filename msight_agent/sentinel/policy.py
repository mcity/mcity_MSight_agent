"""What the investigator may touch: read-only tools only, and no secrets.

Kept free of claude_agent_sdk imports so the agent's test suite can check it.
The SDK session also runs with permission_mode="dontAsk" and only these tools
enabled; check_tool_call is the PreToolUse hook's second line of defence.
"""
import re
from typing import Optional

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
