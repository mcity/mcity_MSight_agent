"""
Shared pytest configuration for mcp_layer unit tests.

Adds the project root and mcp_layer/ to sys.path so that:
  - config.config is importable (used by validate_workflow_state)
  - validate_workflow_state, chat_pipeline, etc. are importable with flat import names
"""
import atexit
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]   # .../mcity_data_agent
_MCP_LAYER  = _REPO_ROOT / "mcp_layer"

for _p in (_REPO_ROOT, _MCP_LAYER):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

_CONFIG_PATH = _REPO_ROOT / "config" / "config.py"


@pytest.fixture(autouse=True)
def _never_write_config_py(monkeypatch):
    """Make config/config.py writes a no-op for the duration of each test.

    WorkflowState.save() rewrites the WORKFLOW_STATE line in the real repo file,
    and several MCP tools rewrite other parts of it. A test that reaches one of
    those paths silently dirties the working tree. Tests that check persistence
    patch save() themselves.

    This covers the writes that go through Path.write_text, which is every
    writer in mcp_layer today. It is a best-effort guard, not a guarantee --
    _restore_config_py below is what actually keeps the tree clean.
    """
    real_write_text = Path.write_text

    def guarded(self, data, *args, **kwargs):
        if self.resolve() == _CONFIG_PATH:
            return len(data)
        return real_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", guarded)


# Snapshot taken in the module body, which pytest executes BEFORE it collects
# any test module. That ordering matters: importing a test module can already
# rewrite WORKFLOW_STATE, which is earlier than any fixture — even a
# session-scoped autouse one — can run.
_CONFIG_SNAPSHOT = _CONFIG_PATH.read_bytes() if _CONFIG_PATH.exists() else None


def _restore_config_py() -> None:
    """Put config/config.py back exactly as it was before collection started.

    The per-test guard above cannot see a write that bypasses Path.write_text,
    and cannot see one that happens at import time either. WorkflowState.load()
    is one such writer: its stale-flag TTL is keyed on the config.py mtime, so
    whether it rewrites the file depends on how old the checkout is rather than
    on which tests ran. Writing the snapshot back is independent of HOW and WHEN
    the file changed, so the working tree ends up clean either way.

    Registered with atexit as well as the sessionfinish hook, because a write
    can still land after pytest's own teardown has finished.
    """
    if _CONFIG_SNAPSHOT is None:
        return
    try:
        if _CONFIG_PATH.read_bytes() != _CONFIG_SNAPSHOT:
            _CONFIG_PATH.write_bytes(_CONFIG_SNAPSHOT)
    except OSError:
        pass


atexit.register(_restore_config_py)


def pytest_sessionfinish(session, exitstatus):
    _restore_config_py()
