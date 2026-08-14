import asyncio
import json
import logging
import os
import re
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastmcp import Client
from fastmcp.client.transports import SSETransport

sys.path.append(os.path.dirname(__file__))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

from chat_pipeline import ChatPipeline
from host_utils import is_cloud_deployment, resolve_host
from llm_clients import ClaudeClient, GeminiClient, GroqClient, OpenAIClient
from mcptools.msight_docker import (
    CALIBRATION_INTRINSICS_REL, CALIBRATION_LOCMAP_REL,
    _calibration_status, _get_msight_path, calibration_state_label,
)
from mcptools.msight_record_archive import (
    DEFAULT_SENSOR_NAME as MSIGHT_DEFAULT_SENSOR_NAME,
    DOWNLOAD_DIR as MSIGHT_DOWNLOAD_DIR,
    _active_sensor_name,
    recording_segment_status,
)
from progress_relay import get_active_progress_cb
from tool_schema import tools
from validate_workflow_state import (
    LabelingBackend, AutoLabelingPhase, LabelingPath, WorkflowState, WORKFLOW_SPECS,
)

load_dotenv()

_LLM_PROVIDERS = {
    "openai": OpenAIClient,
    "groq": GroqClient,
    "gemini": GeminiClient,
    "claude": ClaudeClient,
    "anthropic": ClaudeClient,  # alias
}
_PROVIDER_ENV_VAR = {
    "openai": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
}

llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
if llm_provider not in _LLM_PROVIDERS:
    logging.warning(
        f"[STARTUP] Unknown LLM_PROVIDER '{llm_provider}'. "
        f"Valid values: {list(_LLM_PROVIDERS)}. Falling back to 'openai'."
    )
    llm_provider = "openai"


def _canonical_provider(key: str) -> str:
    key = (key or "").lower()
    return "claude" if key == "anthropic" else key


def available_providers() -> list[str]:
    """Providers whose API key env var is actually set on this server."""
    return [p for p, env in _PROVIDER_ENV_VAR.items() if os.getenv(env, "").strip()]


def get_llm_client(app: FastAPI, requested: str):
    """Resolve + lazily construct + cache a client, never for a provider whose key is missing."""
    avail = available_providers()
    key = _canonical_provider(requested)
    if key not in avail:
        default_key = _canonical_provider(llm_provider)
        key = default_key if default_key in avail else (avail[0] if avail else "openai")
    cache = app.state.llm_clients
    if key not in cache:
        cache[key] = _LLM_PROVIDERS[key]()
    return cache[key]

def _reset_state_on_startup() -> None:
    """Reset persisted state on startup so each server launch begins clean."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from validate_workflow_state import WorkflowState
        WorkflowState().save()
        logging.warning("[STARTUP] Reset WORKFLOW_STATE to defaults.")
    except Exception as e:
        logging.warning(f"[STARTUP] Could not reset WORKFLOW_STATE: {e}")


async def _mcp_log_handler(params) -> None:
    """Relay ctx.log() notifications from a running MCP tool call to whichever
    /chat/stream request is awaiting one, via progress_relay, as an SSE "log" event."""
    cb = get_active_progress_cb()
    if not cb:
        return
    text = params.data if isinstance(params.data, str) else str(params.data)
    await cb("log", {"line": text})


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _reset_state_on_startup()
    app.state.llm_clients = {}
    host = resolve_host()
    mcp_transport = SSETransport(url=f"http://{host}:8000/sse")
    # One persistent MCP connection for the app's lifetime, instead of opening
    # a fresh SSE connection per chat turn (was adding several seconds of
    # connect + initialize overhead to every request that called a tool).
    #
    # deploy-agent.yml launches mcp_server.py and chat_server.py back-to-back
    # with no ordering guarantee, so mcp_server may not be listening yet on
    # the first attempt -- retry with backoff instead of failing startup.
    mcp_client_cm = Client(mcp_transport, log_handler=_mcp_log_handler)
    last_exc: Exception | None = None
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

# Workflow registration lives in validate_workflow_state.WORKFLOW_SPECS — add an
# entry there + drop the file in prompts/workflows/ to register a new workflow.
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

WORKFLOW_PROMPT_TEXT = {
    name: (_PROMPTS_DIR / "workflows" / spec.prompt_file).read_text()
    for name, spec in WORKFLOW_SPECS.items()
}

# Demo mode's default video source is deployment-specific, so it's read from
# .env rather than hardcoded in the prompt file.
_MSIGHT_DEMO_VIDEO_PATH = os.environ.get("MSIGHT_DEMO_VIDEO_PATH", "").strip()
if _MSIGHT_DEMO_VIDEO_PATH:
    _DEMO_VIDEO_HINT = (
        f"The built-in sample video is the fixed default for Demo mode, "
        f"always: video_input='{_MSIGHT_DEMO_VIDEO_PATH}'. Demo takes zero "
        f"input from the user, ever — explain what's about to run (this "
        f"default source, that it's the demo calibration) and call "
        f"start_msight_pipeline with it in the same reply. Do NOT ask the "
        f"user to confirm or choose first — Demo has no consent exchange "
        f"(see STEP 3); asking here just reintroduces the round trip that "
        f"removes."
    )
else:
    _DEMO_VIDEO_HINT = (
        "No default demo video is configured on this host "
        "(MSIGHT_DEMO_VIDEO_PATH is not set in .env) — Demo mode can't run "
        "right now. Tell the user plainly that Demo isn't available on this "
        "host (misconfiguration, not something they can fix), and offer "
        "\"run your own pipeline\" instead. Do not ask them for a source "
        "within Demo — Demo never takes one."
    )

# Checked once at startup (IMDSv2) -- a local file/folder path is only a
# meaningful thing to offer the user when they actually have access to this
# host's filesystem, which isn't true for a cloud sandbox.
_ENVIRONMENT_HINT = (
    "This host is a cloud sandbox — the user has no access to its filesystem. "
    "Never offer or accept a local file/folder path as a video source here; "
    "only an rtsp:// URL or the 📤 upload button are valid ways for this user "
    "to specify their own source."
    if is_cloud_deployment() else
    "This host is local/on-prem — the user may have direct filesystem access, "
    "so a local file/folder path is a valid video source option here."
)
def _strip_marked_block(text: str, tag: str, keep: bool) -> str:
    """Removes a <!-- TAG:START -->...<!-- TAG:END --> block from
    msight_pipeline.txt. keep=True strips just the marker lines; keep=False
    strips the marker lines and everything between them."""
    start, end = f"<!-- {tag}:START -->", f"<!-- {tag}:END -->"
    if keep:
        return text.replace(start + "\n", "").replace(end + "\n", "")
    return re.sub(re.escape(start) + r".*?" + re.escape(end) + r"\n?", "", text, flags=re.DOTALL)

_MSIGHT_PIPELINE_PROMPT_BY_MODE: dict[str, str] = {}
if "msight_pipeline" in WORKFLOW_PROMPT_TEXT:
    _msight_base_text = (
        WORKFLOW_PROMPT_TEXT["msight_pipeline"]
        .replace("{DEMO_VIDEO_HINT}", _DEMO_VIDEO_HINT)
        .replace("{ENVIRONMENT_HINT}", _ENVIRONMENT_HINT)
    )
    _both = _strip_marked_block(_strip_marked_block(_msight_base_text, "STEP2A", keep=True), "STEP2B", keep=True)
    _demo_only = _strip_marked_block(_strip_marked_block(_msight_base_text, "STEP2A", keep=True), "STEP2B", keep=False)
    _custom_only = _strip_marked_block(_strip_marked_block(_msight_base_text, "STEP2B", keep=True), "STEP2A", keep=False)
    _MSIGHT_PIPELINE_PROMPT_BY_MODE = {"": _both, "demo": _demo_only, "custom": _custom_only}
    WORKFLOW_PROMPT_TEXT["msight_pipeline"] = _both

_WORKFLOW_LIST = "\n".join(
    f"{i + 1}. {spec.label} (internal name: {name})"
    for i, (name, spec) in enumerate(WORKFLOW_SPECS.items())
)

BASE_PROMPT = (
    (_PROMPTS_DIR / "base_prompt.txt").read_text().replace("{WORKFLOW_LIST}", _WORKFLOW_LIST)
)


def _load_state_hints() -> dict[str, str]:
    """Parse prompts/state_hints.txt into {name: template}, sections delimited
    by "=== name ===" marker lines. Filled in via .format() in _build_state_hint."""
    text = (_PROMPTS_DIR / "state_hints.txt").read_text()
    sections = re.split(r"^=== (\w+) ===\s*$", text, flags=re.MULTILINE)
    # re.split with a capturing group returns [preamble, name1, body1, name2, body2, ...]
    return {
        name: body.strip()
        for name, body in zip(sections[1::2], sections[2::2])
    }


STATE_HINTS = _load_state_hints()


def _build_system_prompt(state) -> str:
    """Base rules + the active workflow's prompt, if one is selected and registered."""
    if not state or state.workflow_name not in WORKFLOW_PROMPT_TEXT:
        return BASE_PROMPT
    if state.workflow_name == "msight_pipeline" and _MSIGHT_PIPELINE_PROMPT_BY_MODE:
        mode = state.msight_pipeline.mode if state.msight_pipeline else ""
        workflow_text = _MSIGHT_PIPELINE_PROMPT_BY_MODE.get(mode, _MSIGHT_PIPELINE_PROMPT_BY_MODE[""])
    else:
        workflow_text = WORKFLOW_PROMPT_TEXT[state.workflow_name]
    return BASE_PROMPT + "\n\n" + workflow_text


def _attach_source(message: str, source: str | None) -> str:
    """Append a [source: ...] tag when the LLM supplied one, leave message untouched otherwise."""
    if source and source.strip():
        return f"{message.strip()}\n[source: {source.strip()}]"
    return message


def filter_tools_for_state(all_tools: list, state) -> list:
    """Return tools valid for the current step; fails open on None state or exceptions."""
    if state is None:
        return all_tools
    try:
        valid_names = state.valid_tool_names()
    except Exception:
        logging.warning("[FILTER] valid_tool_names() raised — returning full tool list")
        return all_tools
    if valid_names is None:
        return all_tools
    return [t for t in all_tools if t["function"]["name"] in valid_names]


def _msight_calibration_hint() -> str:
    """Live filesystem+checksum check, called directly (not via MCP round trip)
    since it must run on every turn's state hint, not only when the LLM asks."""
    msight_path, err = _get_msight_path()
    if err:
        return "calibration=unknown (MSIGHT_VISION_PATH not configured)"
    return calibration_state_label(_calibration_status(msight_path)["state"], prefixed=True)


def _msight_pipeline_state_hint(mp) -> str:
    """Status for msight_pipeline's customize checklist, reported every turn
    so the LLM never has to infer checklist progress from conversation memory."""
    if mp is None:
        return (
            "msight_checklist(mode=not set, source=not set, "
            f"{_msight_calibration_hint()}, "
            "recording=not active, archiving=not active)"
        )
    if mp.rtsp_url:
        source = f"source=rtsp_url:{mp.rtsp_url}"
    elif mp.video_input:
        source = f"source=video_input:{mp.video_input}"
    else:
        source = "source=not set"
    parts = [f"mode={mp.mode or 'not set'}", source, _msight_calibration_hint()]
    if mp.sensor_name:
        parts.append(f"sensor_name={mp.sensor_name}")
    parts.append(f"recording={'active' if mp.recording_active else 'pending (will auto-start when pipeline starts)' if mp.recording_pending else 'not active'}")
    parts.append(f"archiving={'active' if mp.archiving_active else 'pending (will auto-start when pipeline starts)' if mp.archiving_pending else 'not active'}")
    parts.append(f"pipeline_running={mp.pipeline_running}")
    checklist = "msight_checklist(" + ", ".join(parts) + ")"
    if mp.run_awaiting_confirmation and not mp.run_confirmed:
        return checklist + " | " + STATE_HINTS["msight_run_awaiting_confirm"]
    if mp.pipeline_running:
        source_label = f"rtsp_url:{mp.rtsp_url}" if mp.rtsp_url else f"video_input:{mp.video_input}"
        return checklist + " | " + STATE_HINTS["msight_pipeline_running"].format(source=source_label)
    return checklist


def _build_state_hint(state=None) -> str:
    """Return SESSION_STATE string injected before each user message."""
    try:
        if state is None:
            state = WorkflowState.load()
        if not state.workflow_name:
            return ""
        parts = [f"workflow={state.workflow_name}"]
        spec = WORKFLOW_SPECS.get(state.workflow_name)
        if spec and not spec.requires_dataset:
            if state.workflow_name == "msight_pipeline":
                parts.append(_msight_pipeline_state_hint(state.msight_pipeline))
            return "SESSION_STATE: " + " | ".join(parts)
        if state.dataset_confirmed and state.dataset_name:
            parts.append(f"dataset={state.dataset_name}")
        else:
            parts.append(STATE_HINTS["dataset_not_confirmed"])
        al = state.auto_labeling
        if al:
            if al.labeling_backend == LabelingBackend.BOTH:
                parts.append(STATE_HINTS["backend_both"])
            elif al.labeling_backend and not al.labeling_path:
                parts.append(
                    STATE_HINTS["backend_confirmed_no_path"].format(backend=al.labeling_backend)
                )
            elif al.labeling_backend:
                parts.append(f"backend={al.labeling_backend}")
            if al.labeling_path:
                if al.labeling_path == LabelingPath.MANUAL and not al.manual_classes:
                    export_fn = (
                        "export_to_label_studio"
                        if al.labeling_backend == LabelingBackend.LABEL_STUDIO
                        else "export_to_cvat"
                    )
                    parts.append(
                        STATE_HINTS["manual_no_classes"].format(
                            export_fn=export_fn, dataset_name=state.dataset_name or "?"
                        )
                    )
                else:
                    parts.append(f"labeling_path={al.labeling_path}")
            if (al.manual_classes and not al.cvat_task_id and not al.ls_task_ids
                    and not al.phase):
                classes_s = ", ".join(al.manual_classes)
                if al.export_confirmed:
                    parts.append(
                        STATE_HINTS["manual_classes_export_confirmed"].format(classes=classes_s)
                    )
                else:
                    parts.append(
                        STATE_HINTS["manual_classes_awaiting_confirm"].format(classes=classes_s)
                    )
            if al.phase in (AutoLabelingPhase.ANNOTATING, AutoLabelingPhase.TRAINING):
                # COMPLETE deliberately excluded: it's handled below by the
                # labels_imported check instead. Including it here used to make
                # both hints fire at once with contradictory advice.
                action = {
                    AutoLabelingPhase.ANNOTATING: "export complete — awaiting annotation",
                    AutoLabelingPhase.TRAINING:   "auto-labeling complete — awaiting import",
                }[al.phase]
                parts.append(
                    STATE_HINTS["phase_locked"].format(
                        phase=al.phase, action=action, workflow_name=state.workflow_name
                    )
                )
            if al.models_listed and not al.model_configured:
                parts.append(STATE_HINTS["models_listed_not_configured"])
            if al.model_configured:
                if not al.auto_labeling_complete:
                    if al.run_awaiting_confirmation and not al.run_confirmed:
                        parts.append(STATE_HINTS["model_configured_awaiting_run_confirm"])
                    else:
                        parts.append(STATE_HINTS["model_configured_ready"])
                else:
                    parts.append("model=configured")
            if al.auto_labeling_complete:
                parts.append("auto_labeling=complete")
            if al.labeling_path == LabelingPath.AUTO:
                parts.append(f"localization={'enabled' if al.localization_enabled else 'not configured'}")
            if al.labels_imported:
                parts.append(
                    STATE_HINTS["workflow_complete"].format(
                        labeled_dataset_name=state.labeled_dataset_name
                    )
                )
            if not al.phase and not al.auto_labeling_complete and (al.labeling_path or al.model_configured):
                parts.append(
                    STATE_HINTS["reconfigurable_zone_a"].format(workflow_name=state.workflow_name)
                )
        return "SESSION_STATE: " + " | ".join(parts)
    except Exception:
        return ""


@app.get("/chat/providers")
async def chat_providers():
    return {"providers": available_providers(), "default": "openai"}


@app.post("/msight/upload_calibration")
async def msight_upload_calibration(
    intrinsics: UploadFile = File(...),
    locmap: UploadFile = File(...),
):
    """Writes the user's calibration files into MSight_Vision at the fixed
    paths rfdetr_config.yaml already points at, so the config never needs touching."""
    msight_path, err = _get_msight_path()
    if err:
        return JSONResponse({"status": "error", "message": err}, status_code=400)

    intrinsics_bytes = await intrinsics.read()
    try:
        intrinsics_data = json.loads(intrinsics_bytes)
    except json.JSONDecodeError as e:
        return JSONResponse(
            {"status": "error", "message": f"'{intrinsics.filename}' is not valid JSON: {e}"},
            status_code=400,
        )

    missing_keys = {"f", "x0", "y0"} - set(intrinsics_data.keys())
    if missing_keys:
        return JSONResponse(
            {
                "status": "error",
                "message": f"'{intrinsics.filename}' is missing required key(s): {sorted(missing_keys)}.",
            },
            status_code=400,
        )

    locmap_bytes = await locmap.read()
    try:
        import io
        import numpy as np
        locmap_data = np.load(io.BytesIO(locmap_bytes))
        locmap_keys = set(locmap_data.keys())
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": f"Could not read '{locmap.filename}' as a .npz file: {e}"},
            status_code=400,
        )
    if "x_map" not in locmap_keys or "y_map" not in locmap_keys:
        if "lat_map" in locmap_keys or "lon_map" in locmap_keys:
            return JSONResponse(
                {
                    "status": "error",
                    "message": (
                        f"'{locmap.filename}' has lat_map/lon_map keys — that's Auto Labeling's "
                        "localization format, not the live pipeline's. The live pipeline needs a "
                        ".npz with x_map/y_map keys instead."
                    ),
                },
                status_code=400,
            )
        return JSONResponse(
            {
                "status": "error",
                "message": f"'{locmap.filename}' is missing required key(s) x_map/y_map (found: {sorted(locmap_keys)}).",
            },
            status_code=400,
        )

    intrinsics_dest = msight_path / CALIBRATION_INTRINSICS_REL
    locmap_dest = msight_path / CALIBRATION_LOCMAP_REL
    intrinsics_dest.parent.mkdir(parents=True, exist_ok=True)
    locmap_dest.parent.mkdir(parents=True, exist_ok=True)
    intrinsics_dest.write_bytes(intrinsics_bytes)
    locmap_dest.write_bytes(locmap_bytes)

    return JSONResponse({
        "status": "ok",
        "message": "Calibration files uploaded and applied — they'll be used the next time the pipeline starts.",
        **_calibration_status(msight_path),
    })


# Lets a browser-uploaded video land on this server's disk for use as video_input (cloud sandboxes have no shared host filesystem with the user).
MSIGHT_UPLOAD_DIR = Path("output/msight_uploads")
MSIGHT_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
MSIGHT_MAX_UPLOAD_BYTES = int(os.environ.get("MSIGHT_MAX_UPLOAD_BYTES", 2 * 1024**3))  # 2GB default
_UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1MB


@app.post("/msight/upload_video")
async def msight_upload_video(video: UploadFile = File(...)):
    """Streams the upload to disk in chunks (unlike upload_calibration's small-file `read()`) and enforces MSIGHT_MAX_UPLOAD_BYTES while streaming."""
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in MSIGHT_VIDEO_EXTENSIONS:
        return JSONResponse(
            {
                "status": "error",
                "message": (
                    f"Unsupported file type '{suffix or '(none)'}' — expected one of "
                    f"{sorted(MSIGHT_VIDEO_EXTENSIONS)}."
                ),
            },
            status_code=400,
        )

    MSIGHT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = MSIGHT_UPLOAD_DIR / f"upload_{datetime.now().strftime('%Y%m%dT%H%M%S%f')}{suffix}"

    written = 0
    try:
        with open(dest, "wb") as f:
            while chunk := await video.read(_UPLOAD_CHUNK_SIZE):
                written += len(chunk)
                if written > MSIGHT_MAX_UPLOAD_BYTES:
                    raise ValueError("size_limit_exceeded")
                f.write(chunk)
    except ValueError:
        dest.unlink(missing_ok=True)
        limit_gb = MSIGHT_MAX_UPLOAD_BYTES / 1024**3
        return JSONResponse(
            {"status": "error", "message": f"Upload exceeds the {limit_gb:.1f}GB limit for this sandbox."},
            status_code=413,
        )
    except Exception as e:
        dest.unlink(missing_ok=True)
        return JSONResponse({"status": "error", "message": f"Upload failed: {e}"}, status_code=500)

    return JSONResponse({
        "status": "ok",
        "message": f"Uploaded '{video.filename}' ({written / 1024**2:.1f} MB).",
        # Absolute, since docker compose resolves VIDEO_INPUT relative to its own cwd, not this server's.
        "video_input": str(dest.resolve()),
    })


@app.get("/msight/download_recording/{filename}")
async def msight_download_recording(filename: str):
    """Serves a finished recording as a browser download, since a chat reply
    can only hand back a server-local path the user may have no access to."""
    # Reject anything but a bare filename so a crafted "../.." can't escape the dir.
    if "/" in filename or "\\" in filename or filename in (".", ".."):
        return JSONResponse({"status": "error", "message": "Invalid filename."}, status_code=400)

    path = (MSIGHT_DOWNLOAD_DIR / filename).resolve()
    if MSIGHT_DOWNLOAD_DIR.resolve() not in path.parents or not path.is_file():
        return JSONResponse({"status": "error", "message": "Recording not found."}, status_code=404)

    return FileResponse(path, media_type="video/mp4", filename=filename)


@app.get("/msight/record_status")
async def msight_record_status():
    """Live recording-progress poll for the UI's status badge, independent of the chat/LLM loop."""
    state = WorkflowState.load()
    mp = state.msight_pipeline
    if mp is None or not (mp.recording_active or mp.recording_pending):
        return JSONResponse({
            "status": "ok",
            "recording_active": False,
            "recording_pending": False,
            "segment_count": 0,
            "seconds_since_last": None,
            "message": "Not recording.",
        })

    if mp.recording_pending:
        return JSONResponse({
            "status": "ok",
            "recording_active": False,
            "recording_pending": True,
            "segment_count": 0,
            "seconds_since_last": None,
            "message": "Recording will start automatically once the pipeline is running.",
        })

    msight_path, path_err = _get_msight_path()
    if mp.sensor_name:
        sensor = mp.sensor_name
    elif not path_err:
        sensor = _active_sensor_name(msight_path)
    else:
        sensor = MSIGHT_DEFAULT_SENSOR_NAME
    seg = recording_segment_status(sensor)

    if seg["segment_count"] == 0:
        message = "Recording — capturing first clip, nothing saved yet."
    else:
        plural = "s" if seg["segment_count"] != 1 else ""
        message = (
            f"Recording — {seg['segment_count']} clip{plural} captured "
            f"(most recent {seg['seconds_since_last']}s ago). Stop recording to download."
        )

    return JSONResponse({
        "status": "ok",
        "recording_active": True,
        "recording_pending": False,
        **seg,
        "message": message,
    })


@app.post("/chat/stream")
async def chat_stream(request: Request):
    """SSE endpoint. Events: status, log, progress, reply (terminal), error."""
    data    = await request.json()
    message = data.get("message", "")
    history = data.get("history", [])
    llm_client = get_llm_client(request.app, data.get("provider", ""))

    MAX_HISTORY_TURNS = 4
    if len(history) > MAX_HISTORY_TURNS:
        history = history[-MAX_HISTORY_TURNS:]

    try:
        _state = WorkflowState.load()
    except Exception:
        _state = None

    messages = [{"role": "system", "content": _build_system_prompt(_state)}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user",      "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    if _state:
        al = _state.auto_labeling
        logging.warning(
            f"[STREAM STATE] loaded: workflow={_state.workflow_name!r} "
            f"dataset={_state.dataset_name!r} confirmed={_state.dataset_confirmed} "
            f"backend={al.labeling_backend if al else ''!r} "
            f"path={al.labeling_path if al else ''!r} "
            f"phase={al.phase if al else ''!r}"
        )

    if _state and _state.workflow_just_reset:
        messages.append({"role": "system", "content": (
            "WORKFLOW_RESET: The previous workflow session has completely ended. "
            "All parameters (dataset, backend, model, classes) have been cleared. "
            "The conversation history above belongs to a DIFFERENT session — "
            "do NOT reuse any dataset name, backend, model, or configuration from it. "
            "The user must explicitly provide all values from scratch in this new session."
        )})
        _state.workflow_just_reset = False
        _state.save()

    state_hint = _build_state_hint(_state)
    if state_hint:
        messages.append({"role": "system", "content": state_hint})
        logging.warning(f"[STREAM HINT] {state_hint}")
    messages.append({"role": "user", "content": message})

    active_tools = filter_tools_for_state(tools, _state)
    logging.warning(
        f"[STREAM TOOLS] Active ({len(active_tools)}): "
        f"{[t['function']['name'] for t in active_tools]}"
    )

    event_queue: asyncio.Queue = asyncio.Queue()

    async def progress_cb(event_type: str, evt_data: dict) -> None:
        await event_queue.put((event_type, evt_data))

    async def run_pipeline() -> None:
        try:
            current_tools = active_tools
            pipeline = None
            MAX_AGENTIC_ITERATIONS = 5

            for iteration in range(MAX_AGENTIC_ITERATIONS):
                tool_choice = "required" if iteration == 0 else "auto"

                try:
                    assistant_message = await llm_client.chat(
                        messages, tools=current_tools, tool_choice=tool_choice
                    )
                except Exception as e:
                    err = str(e).lower()
                    msg = (
                        "The request timed out reaching the AI service. Please try again."
                        if "timeout" in err or "connecttimeout" in err
                        else "Something went wrong connecting to the AI service. Please try again."
                    )
                    await event_queue.put(("error", {"message": msg}))
                    return

                if not (hasattr(assistant_message, "tool_calls") and assistant_message.tool_calls):
                    reply = assistant_message.content or ""
                    logging.warning(f"[STREAM DECISION] iter={iteration} → end_turn (no tool call)")
                    await event_queue.put(("reply", {"message": reply}))
                    return

                tool_calls = assistant_message.tool_calls
                logging.warning(
                    f"[STREAM DECISION] iter={iteration} tool_choice={tool_choice!r} → "
                    f"{[f'{c.function.name}({c.function.arguments[:60]})' for c in tool_calls]}"
                )

                if len(tool_calls) == 1 and tool_calls[0].function.name == "send_reply":
                    try:
                        args  = json.loads(tool_calls[0].function.arguments)
                        reply = _attach_source(args.get("message", ""), args.get("source"))
                    except Exception:
                        reply = "Something went wrong. Please try again."
                    await event_queue.put(("reply", {"message": reply}))
                    return

                messages.append({
                    "role":       "assistant",
                    "content":    assistant_message.content or "",
                    "tool_calls": [
                        {
                            "id":       c.id,
                            "type":     "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments},
                        }
                        for c in tool_calls
                    ],
                })

                if pipeline is None:
                    pipeline = ChatPipeline(mcp_client=request.app.state.mcp_client, llm=llm_client)

                tool_results, early_reply = await pipeline.run(
                    tool_calls, messages, progress_cb=progress_cb
                )

                if early_reply is not None:
                    await event_queue.put(("reply", {"message": early_reply}))
                    return

                if iteration > 0:
                    logging.warning(
                        f"[STREAM] Agentic loop: iteration {iteration + 1} — "
                        f"tools called: {[r['name'] for r in tool_results]}"
                    )

                current_tools = filter_tools_for_state(tools, pipeline.state)

                # Rebuild so a workflow change earlier this iteration (e.g.
                # select_workflow) is reflected for the rest of the turn --
                # needed by workflows that fall through instead of HardStop
                # right after selection (e.g. msight_pipeline).
                messages[0]["content"] = _build_system_prompt(pipeline.state)

                workflow_spec = WORKFLOW_SPECS.get(pipeline.state.workflow_name)
                tools_called = [r["name"] for r in tool_results]
                if (
                    pipeline.state.workflow_name
                    and workflow_spec is not None and workflow_spec.requires_dataset
                    and not pipeline.state.dataset_confirmed
                    and "list_datasets" not in tools_called
                ):
                    messages.append({
                        "role":    "system",
                        "content": (
                            "Reminder: the user has selected a workflow but has not yet "
                            "confirmed a dataset. Show the user the dataset list returned "
                            "by list_datasets above. Do NOT list datasets from memory."
                        ),
                    })

            logging.warning(f"[STREAM] Exceeded {MAX_AGENTIC_ITERATIONS} agentic iterations")
            await event_queue.put(("reply", {"message": "I wasn't able to complete this step. Please try again."}))

        except Exception as e:
            logging.warning(f"[STREAM] run_pipeline exception: {e}")
            await event_queue.put(("error", {"message": "An unexpected error occurred. Please try again."}))
        finally:
            await event_queue.put(None)

    async def generate():
        task = asyncio.create_task(run_pipeline())
        try:
            while True:
                item = await event_queue.get()
                if item is None:
                    break
                event_type, evt_data = item
                yield f"event: {event_type}\ndata: {json.dumps(evt_data)}\n\n"
        finally:
            await task

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)