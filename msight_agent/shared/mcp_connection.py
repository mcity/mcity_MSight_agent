"""The persistent MCP client connection shared by chat_server.py and msight_api.py.

The connection outlives any one request, so it must survive mcp_server.py restarting:
a transport failure reconnects, then either retries (read-only tools) or reports that
the call's outcome is unknown (anything that changes state -- it may already have run).
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastmcp import Client
from fastmcp.client.transports import SSETransport
from fastmcp.exceptions import ClientError

from msight_agent.shared.host_utils import resolve_host

_STARTUP_ATTEMPTS = 15
_RECONNECT_ATTEMPTS = 3
_RETRY_SECONDS = 2
_PING_TIMEOUT = 5
# Safe to send again after a dropped connection: they only read.
_READ_ONLY_PREFIXES = ("get_", "list_", "check_", "diagnose_", "inspect_")


class MCPUnavailable(Exception):
    """mcp_server.py can't be reached, even after reconnecting."""


class MCPCallInterrupted(Exception):
    """The connection dropped during a state-changing call: it may or may not have run."""


class MCPConnection:
    """A fastmcp Client that reconnects. Same call_tool() interface as Client."""

    def __init__(self, log_handler=None):
        self._log_handler = log_handler
        self._client_cm = None
        self._client = None
        self._lock = asyncio.Lock()

    async def connect(self, attempts: int) -> None:
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            client_cm = Client(SSETransport(url=f"http://{resolve_host()}:8000/sse"),
                               log_handler=self._log_handler)
            try:
                self._client = await client_cm.__aenter__()
                self._client_cm = client_cm
                return
            except Exception as e:
                last_exc = e
                logging.warning(f"[MCP] connect attempt {attempt}/{attempts} failed: {e}")
                if attempt < attempts:
                    await asyncio.sleep(_RETRY_SECONDS)
        raise MCPUnavailable(f"could not connect to the MCP server after {attempts} attempts") from last_exc

    async def close(self) -> None:
        client_cm, self._client_cm, self._client = self._client_cm, None, None
        if client_cm is not None:
            try:
                await client_cm.__aexit__(None, None, None)
            except Exception as e:  # already broken; nothing to clean up
                logging.warning(f"[MCP] closing a dead connection: {e}")

    async def _reconnect(self, broken_client) -> None:
        async with self._lock:
            if self._client is not broken_client and self._client is not None:
                return  # another request already reconnected
            await self.close()
            await self.connect(_RECONNECT_ATTEMPTS)
            logging.warning("[MCP] reconnected to the MCP server")

    async def _ensure_live(self) -> None:
        """Before a state-changing call: reconnect now if the server restarted since the
        last call, so a failure during the call itself really means a mid-call drop."""
        client = self._client
        try:
            if client is None:
                raise ConnectionError("not connected")
            await asyncio.wait_for(client.ping(), timeout=_PING_TIMEOUT)
        except Exception as e:
            logging.warning(f"[MCP] connection is dead ({type(e).__name__}: {e}); reconnecting before the call")
            await self._reconnect(client)

    async def call_tool(self, name: str, args: dict | None = None):
        read_only = name.startswith(_READ_ONLY_PREFIXES)
        if not read_only:
            await self._ensure_live()
        client = self._client
        try:
            if client is None:
                raise ConnectionError("not connected")
            return await client.call_tool(name, args or {})
        except ClientError:
            raise  # the tool itself reported an error; the connection is fine
        except Exception as e:
            logging.warning(f"[MCP] {name} failed on the transport ({type(e).__name__}: {e}); reconnecting")
            await self._reconnect(client)  # raises MCPUnavailable if the server is gone
            if read_only:
                return await self._client.call_tool(name, args or {})
            raise MCPCallInterrupted(name) from e


@asynccontextmanager
async def mcp_session(log_handler=None):
    """Connect to mcp_server.py, retrying because it may not be listening yet at startup."""
    connection = MCPConnection(log_handler)
    await connection.connect(_STARTUP_ATTEMPTS)
    try:
        yield connection
    finally:
        await connection.close()
