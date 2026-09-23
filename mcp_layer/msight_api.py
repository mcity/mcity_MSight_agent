"""Plain REST peer to the MCP tool surface -- the `API` front door from the
original architecture sketch, alongside `MCP` (mcp_server.py, reached by the
LLM through chat_server.py). Both are thin front doors onto the same
MSightControlPlane; this one is just triggered by a plain HTTP request
instead of a model's tool-call decision.

Every route below is a call into an existing MCP tool over the same kind of
persistent MCP client connection chat_server.py already opens in its own
lifespan -- deliberately never importing mcptools.* directly. MSightControlPlane's
node tracking is in-memory, inside whichever process constructed it (that's
mcp_server.py's process today); a second instance constructed here would
silently diverge from the one that's actually tracking running nodes.
"""
import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastmcp import Client
from fastmcp.client.transports import SSETransport

sys.path.append(os.path.dirname(__file__))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

from host_utils import resolve_host


@asynccontextmanager
async def _lifespan(app: FastAPI):
    host = resolve_host()
    mcp_transport = SSETransport(url=f"http://{host}:8000/sse")
    # Same retry-with-backoff as chat_server.py's lifespan -- deploy-agent.yml
    # launches services with no ordering guarantee, so mcp_server.py may not
    # be listening yet on the first attempt.
    mcp_client_cm = Client(mcp_transport)
    last_exc: Optional[Exception] = None
    for attempt in range(15):
        try:
            mcp_client = await mcp_client_cm.__aenter__()
            break
        except Exception as e:
            last_exc = e
            logging.warning(
                f"[STARTUP] MCP connect attempt {attempt + 1}/15 failed: {e}; retrying in 2s"
            )
            await asyncio.sleep(2)
    else:
        raise RuntimeError("Could not connect to MCP server after 15 attempts") from last_exc

    app.state.mcp_client = mcp_client
    try:
        yield
    finally:
        await mcp_client_cm.__aexit__(None, None, None)


app = FastAPI(lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _json_body(request: Request) -> dict:
    body = await request.body()
    if not body:
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {}


def _unwrap_json(raw) -> str:
    """Like pipeline_common.unwrap_tool_output, minus its \\n -> real-newline
    substitution -- that substitution is meant to make tool output read
    nicely in the chat UI, but it corrupts JSON re-parsing here whenever a
    string value contains a real newline (confirmed: get_msight_logs's
    `logs` field always does), since a raw newline inside a JSON string is
    invalid syntax."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if hasattr(raw, "text"):
        return (raw.text or "").strip()
    if isinstance(raw, list):
        parts = [_unwrap_json(x) for x in raw]
        return "\n".join(p for p in parts if p).strip()
    if isinstance(raw, dict):
        if "text" in raw and isinstance(raw["text"], str):
            return raw["text"].strip()
        if "content" in raw and isinstance(raw["content"], list):
            return _unwrap_json(raw["content"])
    return str(raw).strip()


async def _call_tool(request: Request, name: str, args: dict) -> JSONResponse:
    raw = await request.app.state.mcp_client.call_tool(name, args)
    text = _unwrap_json(raw)
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return JSONResponse({"status": "error", "message": text}, status_code=502)
    status_code = 200 if data.get("status") == "ok" else 502
    return JSONResponse(data, status_code=status_code)


@app.get("/status")
async def get_status(request: Request):
    return await _call_tool(request, "get_msight_status", {})


@app.get("/logs")
async def get_logs(request: Request, service: Optional[str] = None, tail: int = 200):
    args = {"tail": tail}
    if service:
        args["service"] = service
    return await _call_tool(request, "get_msight_logs", args)


@app.get("/calibration")
async def get_calibration(request: Request):
    return await _call_tool(request, "check_msight_calibration_status", {})


@app.post("/pipeline/start")
async def pipeline_start(request: Request):
    data = await _json_body(request)
    args = {
        "video_input": data.get("video_input"),
        "rtsp_url": data.get("rtsp_url"),
        "sensor_name": data.get("sensor_name"),
        "build": data.get("build", False),
    }
    return await _call_tool(request, "start_msight_pipeline", args)


@app.post("/pipeline/stop")
async def pipeline_stop(request: Request):
    data = await _json_body(request)
    args = {"remove_volumes": data.get("remove_volumes", False)}
    return await _call_tool(request, "stop_msight_pipeline", args)


@app.post("/recording/start")
async def recording_start(request: Request):
    data = await _json_body(request)
    args = {"sensor_name": data.get("sensor_name")}
    return await _call_tool(request, "start_msight_recording", args)


@app.post("/recording/stop")
async def recording_stop(request: Request):
    return await _call_tool(request, "stop_msight_recording", {})


@app.post("/archiving/start")
async def archiving_start(request: Request):
    data = await _json_body(request)
    if not data.get("s3_bucket"):
        return JSONResponse({"status": "error", "message": "s3_bucket is required."}, status_code=400)
    args = {"s3_bucket": data["s3_bucket"], "s3_prefix": data.get("s3_prefix")}
    return await _call_tool(request, "start_msight_archiving", args)


@app.post("/archiving/stop")
async def archiving_stop(request: Request):
    return await _call_tool(request, "stop_msight_archiving", {})


@app.get("/record_archive/status")
async def record_archive_status(request: Request):
    return await _call_tool(request, "get_msight_record_archive_status", {})


@app.get("/nodes/types")
async def node_types(request: Request):
    return await _call_tool(request, "list_msight_node_types", {})


@app.post("/nodes/add")
async def node_add(request: Request):
    data = await _json_body(request)
    if not data.get("node_type"):
        return JSONResponse({"status": "error", "message": "node_type is required."}, status_code=400)
    args = {
        "node_type": data["node_type"],
        "name": data.get("name"),
        "config": data.get("config"),
    }
    return await _call_tool(request, "add_msight_node", args)


@app.post("/nodes/remove")
async def node_remove(request: Request):
    data = await _json_body(request)
    if not data.get("name"):
        return JSONResponse({"status": "error", "message": "name is required."}, status_code=400)
    return await _call_tool(request, "remove_msight_node", {"name": data["name"]})


@app.get("/reference/{topic}")
async def reference(request: Request, topic: str):
    return await _call_tool(request, "get_msight_reference", {"topic": topic})


# Dashboard static files (mcp_layer/dashboard, `npm run build`'d into
# dist/spa) -- mounted last, deliberately, so it only catches whatever none
# of the explicit routes above matched (Starlette tries routes in
# registration order; a Mount registered first would shadow everything).
# Fetches from the dashboard to this same origin need no CORS/CSP exception
# in production, unlike the `quasar dev` case (a different port, handled by
# a dev-only <meta> tag in the dashboard's own index.html).
_DASHBOARD_DIST = Path(__file__).resolve().parent / "dashboard" / "dist" / "spa"
if _DASHBOARD_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_DASHBOARD_DIST), html=True), name="dashboard")
else:
    logging.warning(f"[STARTUP] Dashboard build not found at {_DASHBOARD_DIST} -- run 'npm run build' in mcp_layer/dashboard/. Serving API only.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
