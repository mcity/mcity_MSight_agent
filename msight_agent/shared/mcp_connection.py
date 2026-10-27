"""The persistent MCP client connection shared by chat_server.py and msight_api.py."""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastmcp import Client
from fastmcp.client.transports import SSETransport

from msight_agent.shared.host_utils import resolve_host

_CONNECT_ATTEMPTS = 15
_RETRY_SECONDS = 2


@asynccontextmanager
async def mcp_session(log_handler=None):
    """Connect to mcp_server.py, retrying because it may not be listening yet at startup."""
    client_cm = Client(SSETransport(url=f"http://{resolve_host()}:8000/sse"), log_handler=log_handler)
    last_exc: Exception | None = None
    for attempt in range(1, _CONNECT_ATTEMPTS + 1):
        try:
            client = await client_cm.__aenter__()
            break
        except Exception as e:
            last_exc = e
            logging.warning(f"[STARTUP] MCP connect attempt {attempt}/{_CONNECT_ATTEMPTS} failed: {e}; "
                            f"retrying in {_RETRY_SECONDS}s")
            await asyncio.sleep(_RETRY_SECONDS)
    else:
        raise RuntimeError(f"Could not connect to MCP server after {_CONNECT_ATTEMPTS} attempts") from last_exc
    try:
        yield client
    finally:
        await client_cm.__aexit__(None, None, None)
