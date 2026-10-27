"""MCPConnection survives mcp_server.py restarting. Before this, a restart made every
later tool call return an empty result and the model flailed through unrelated tools."""
import asyncio

import pytest
from fastmcp.exceptions import ClientError

from msight_agent.shared import mcp_connection
from msight_agent.shared.mcp_connection import MCPCallInterrupted, MCPConnection, MCPUnavailable


class FakeClient:
    def __init__(self, fail_calls: int = 0, ping_ok: bool = True, error=None):
        self.fail_calls, self.ping_ok, self.error, self.calls = fail_calls, ping_ok, error, []

    async def ping(self):
        if not self.ping_ok:
            raise ConnectionError("dead")
        return True

    async def call_tool(self, name, args):
        self.calls.append(name)
        if self.error:
            raise self.error
        if self.fail_calls:
            self.fail_calls -= 1
            raise RuntimeError("Connection closed")
        return "ok"


def _connection(monkeypatch, first: FakeClient, after_reconnect: FakeClient | None):
    conn = MCPConnection()
    conn._client = first

    async def connect(attempts):
        if after_reconnect is None:
            raise MCPUnavailable("down")
        conn._client = after_reconnect

    async def close():
        conn._client = None

    monkeypatch.setattr(conn, "connect", connect)
    monkeypatch.setattr(conn, "close", close)
    return conn


def test_read_only_call_is_retried_after_reconnecting(monkeypatch):
    fresh = FakeClient()
    conn = _connection(monkeypatch, FakeClient(fail_calls=1), fresh)
    assert asyncio.run(conn.call_tool("get_msight_status", {})) == "ok"
    assert fresh.calls == ["get_msight_status"]


def test_state_changing_call_is_not_resent_after_a_mid_call_drop(monkeypatch):
    fresh = FakeClient()
    conn = _connection(monkeypatch, FakeClient(fail_calls=1), fresh)
    with pytest.raises(MCPCallInterrupted):
        asyncio.run(conn.call_tool("start_msight_pipeline", {}))
    assert fresh.calls == []  # never sent twice


def test_dead_connection_is_replaced_before_a_state_changing_call(monkeypatch):
    fresh = FakeClient()
    stale = FakeClient(ping_ok=False)
    conn = _connection(monkeypatch, stale, fresh)
    assert asyncio.run(conn.call_tool("add_msight_node", {})) == "ok"
    assert stale.calls == [] and fresh.calls == ["add_msight_node"]


def test_unreachable_server_raises_unavailable(monkeypatch):
    conn = _connection(monkeypatch, FakeClient(fail_calls=1), None)
    with pytest.raises(MCPUnavailable):
        asyncio.run(conn.call_tool("get_msight_status", {}))


def test_a_tools_own_error_is_not_a_connection_problem(monkeypatch):
    conn = _connection(monkeypatch, FakeClient(error=ClientError("bad args")), FakeClient())
    with pytest.raises(ClientError):
        asyncio.run(conn.call_tool("get_msight_status", {}))


def test_read_only_prefixes_cover_the_observation_tools():
    for name in ("get_msight_status", "get_msight_logs", "diagnose_msight_pipeline", "inspect_host",
                 "list_msight_node_types", "check_msight_calibration_status", "get_msight_reference"):
        assert name.startswith(mcp_connection._READ_ONLY_PREFIXES)
    for name in ("start_msight_pipeline", "add_msight_node", "remove_msight_node", "export_to_cvat"):
        assert not name.startswith(mcp_connection._READ_ONLY_PREFIXES)
