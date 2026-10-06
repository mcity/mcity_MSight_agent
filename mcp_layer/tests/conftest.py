"""Puts the project root and mcp_layer/ on sys.path for flat imports."""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]   # .../mcity_data_agent
_MCP_LAYER  = _REPO_ROOT / "mcp_layer"

for _p in (_REPO_ROOT, _MCP_LAYER):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
