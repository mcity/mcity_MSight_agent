"""On-demand reference docs, served as a plain tool (the client doesn't use MCP Resources)."""
from pathlib import Path

from mcptools import mcp
from mcptools.mcp_json import error_json, ok_json

_REFERENCE_DIR = Path(__file__).resolve().parents[1] / "prompts" / "msight_reference"

_TOPICS: dict[str, str] = {
    "node_catalog": "node_catalog.md",
    "common_failure_modes": "common_failure_modes.md",
    "diagnosing_stalled_nodes": "diagnosing_stalled_nodes.md",
}


@mcp.tool()
def get_msight_reference(topic: str) -> str:
    if topic not in _TOPICS:
        return error_json(f"Unknown topic '{topic}'. Available: {sorted(_TOPICS)}")
    text = (_REFERENCE_DIR / _TOPICS[topic]).read_text()
    return ok_json(topic=topic, text=text)
