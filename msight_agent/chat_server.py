"""Chat backend (port 8001): the /chat/stream SSE endpoint the web UI talks to,
plus MSight file endpoints. The agent itself lives in msight_agent/agent/."""
import asyncio
import json
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")

from msight_agent.agent.dispatch import ChatPipeline
from msight_agent.agent.guards import filter_tools_for_state
from msight_agent.agent.llm_clients import PROVIDERS, available_providers, default_provider, resolve_provider
from msight_agent.agent.loop import AgentTurn
from msight_agent.agent.prompts import build_system_prompt
from msight_agent.agent.state import WorkflowState
from msight_agent.agent.state_hints import build_state_hint
from msight_agent.agent.tool_schema import tools
from msight_agent.msight import routes as msight_routes
from msight_agent.shared.mcp_connection import mcp_session
from msight_agent.shared.progress_relay import get_active_progress_cb

load_dotenv()

MAX_HISTORY_TURNS = 4
# Keyword check for the default workflow; anything else defaults to msight_pipeline.
_AUTO_LABELING_TRIGGERS = ("auto labeling", "auto-labeling", "auto labelling", "auto-labelling", "autolabeling")
_WORKFLOW_RESET_NOTE = (
    "WORKFLOW_RESET: The previous workflow session has completely ended. "
    "All parameters (dataset, backend, model, classes) have been cleared. "
    "The conversation history above belongs to a DIFFERENT session — "
    "do NOT reuse any dataset name, backend, model, or configuration from it. "
    "The user must explicitly provide all values from scratch in this new session."
)


async def _mcp_log_handler(params) -> None:
    """Relay MCP ctx.log() notifications to the active /chat/stream as SSE "log" events."""
    if cb := get_active_progress_cb():
        await cb("log", {"line": params.data if isinstance(params.data, str) else str(params.data)})


@asynccontextmanager
async def _lifespan(app: FastAPI):
    try:
        WorkflowState().save()  # each server launch begins clean
        logging.warning("[STARTUP] Reset WORKFLOW_STATE to defaults.")
    except Exception as e:
        logging.warning(f"[STARTUP] Could not reset WORKFLOW_STATE: {e}")
    app.state.llm_clients = {}
    async with mcp_session(log_handler=_mcp_log_handler) as client:
        app.state.mcp_client = client
        yield


app = FastAPI(lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(msight_routes.router)


def _llm_client(requested: str):
    """A cached client for the requested provider (or a configured fallback)."""
    name = resolve_provider(requested)
    cache = app.state.llm_clients
    if name not in cache:
        cache[name] = PROVIDERS[name]()
    return cache[name]


async def _select_default_workflow(state: WorkflowState | None, message: str, llm_client, mcp_client
                                   ) -> tuple[WorkflowState | None, ChatPipeline | None]:
    """Pick a workflow in code via ChatPipeline's own handler; no-op if one is set."""
    if state is None or state.workflow_name:
        return state, None
    workflow = ("auto_labeling" if any(t in message.lower() for t in _AUTO_LABELING_TRIGGERS)
                else "msight_pipeline")
    try:
        pipeline = ChatPipeline(mcp_client=mcp_client, llm=llm_client)
        pipeline.state = state
        await pipeline._handle_select_or_switch_workflow("select_workflow", {"workflow_name": workflow}, mcp_client)
        logging.warning(f"[STREAM] Defaulted to workflow={workflow!r} in code (no LLM call)")
        return pipeline.state, pipeline
    except Exception as e:
        logging.warning(f"[STREAM] Default workflow selection failed, leaving to the model's own rule: {e}")
        return state, None


async def _build_messages(state: WorkflowState | None, history: list, message: str) -> list[dict]:
    """System prompt, recent history, this turn's state hint, then the user message."""
    messages = [{"role": "system", "content": build_system_prompt(state)}]
    for user_msg, assistant_msg in history[-MAX_HISTORY_TURNS:]:
        messages += [{"role": "user", "content": user_msg}, {"role": "assistant", "content": assistant_msg}]
    if state and state.workflow_just_reset:
        messages.append({"role": "system", "content": _WORKFLOW_RESET_NOTE})
        state.workflow_just_reset = False
        state.save()
    if state_hint := await build_state_hint(state):
        messages.append({"role": "system", "content": state_hint})
        logging.warning(f"[STREAM HINT] {state_hint}")
    messages.append({"role": "user", "content": message})
    return messages


@app.get("/chat/providers")
async def chat_providers():
    return {"providers": available_providers(), "default": default_provider()}


@app.post("/chat/stream")
async def chat_stream(request: Request):
    """SSE endpoint. Events: status, log, progress, intro, raw_logs, reply (terminal), error."""
    data = await request.json()
    message = data.get("message", "")
    llm_client = _llm_client(data.get("provider", ""))
    mcp_client = request.app.state.mcp_client

    try:
        state = WorkflowState.load()
    except Exception:
        state = None
    state, early_pipeline = await _select_default_workflow(state, message, llm_client, mcp_client)
    if state:
        logging.warning(f"[STREAM STATE] workflow={state.workflow_name!r} dataset={state.dataset_name!r} "
                        f"confirmed={state.dataset_confirmed}")

    active_tools = filter_tools_for_state(tools, state)
    logging.warning(f"[STREAM TOOLS] Active ({len(active_tools)}): {[t['function']['name'] for t in active_tools]}")

    events: asyncio.Queue = asyncio.Queue()

    async def emit(event_type: str, evt_data: dict) -> None:
        await events.put((event_type, evt_data))

    turn = AgentTurn(
        message=message, messages=await _build_messages(state, data.get("history", []), message),
        tools=active_tools, llm_client=llm_client, mcp_client=mcp_client,
        pipeline=early_pipeline, emit=emit,
    )

    async def run_turn() -> None:
        try:
            await turn.run()
        except Exception as e:
            logging.warning(f"[STREAM] Agent turn failed: {e}")
            await emit("error", {"message": "An unexpected error occurred. Please try again."})
        finally:
            await events.put(None)

    async def generate():
        task = asyncio.create_task(run_turn())
        try:
            while (item := await events.get()) is not None:
                event_type, evt_data = item
                yield f"event: {event_type}\ndata: {json.dumps(evt_data)}\n\n"
        finally:
            await task

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
