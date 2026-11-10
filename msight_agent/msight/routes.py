"""MSight endpoints served by chat_server.py for the chat UI: calibration and video
uploads, recording download, the recording-progress badge, and the live incident feed."""
import asyncio
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import redis
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from msight_agent.msight.runtime import incidents
from msight_agent.msight.runtime.calibration import calibration_status, locmap_problem, write_calibration
from msight_agent.msight.runtime.paths import get_msight_path
from msight_agent.msight.runtime.utils import redis_kwargs
from msight_agent.msight.tools.record_archive import (
    DEFAULT_SENSOR_NAME, DOWNLOAD_DIR, active_sensor_name, recording_segment_status,
)
from msight_agent.agent.state import WorkflowState
from msight_agent.shared.paths import MSIGHT_UPLOADS_DIR

router = APIRouter(prefix="/msight")

UPLOAD_DIR = MSIGHT_UPLOADS_DIR
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
MAX_UPLOAD_BYTES = int(os.environ.get("MSIGHT_MAX_UPLOAD_BYTES", 2 * 1024**3))  # 2GB default
_UPLOAD_CHUNK_SIZE = 1024 * 1024


def _error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"status": "error", "message": message}, status_code=status_code)


def _intrinsics_error(name: str, data: bytes) -> str | None:
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as e:
        return f"'{name}' is not valid JSON: {e}"
    missing = {"f", "x0", "y0"} - set(parsed)
    return f"'{name}' is missing required key(s): {sorted(missing)}." if missing else None


def _locmap_error(name: str, data: bytes, intrinsics: bytes) -> str | None:
    try:
        npz = np.load(io.BytesIO(data))
        keys = set(npz.keys())
    except Exception as e:
        return f"Could not read '{name}' as a .npz file: {e}"
    if {"x_map", "y_map"} <= keys:
        k = json.loads(intrinsics)
        problem = locmap_problem(npz["x_map"].shape, npz["y_map"].shape, (k["x0"], k["y0"]))
        return f"'{name}': {problem}" if problem else None
    if keys & {"lat_map", "lon_map"}:
        return (f"'{name}' has lat_map/lon_map keys — that's Auto Labeling's localization format, "
                "not the live pipeline's. The live pipeline needs a .npz with x_map/y_map keys instead.")
    return f"'{name}' is missing required key(s) x_map/y_map (found: {sorted(keys)})."


@router.post("/upload_calibration")
async def upload_calibration(intrinsics: UploadFile = File(...), locmap: UploadFile = File(...)):
    """Write calibration files to the fixed paths rfdetr_config.yaml already references."""
    msight_path, err = get_msight_path()
    if err:
        return _error(err)
    intrinsics_bytes, locmap_bytes = await intrinsics.read(), await locmap.read()
    if err := (_intrinsics_error(intrinsics.filename, intrinsics_bytes)
               or _locmap_error(locmap.filename, locmap_bytes, intrinsics_bytes)):
        return _error(err)
    write_calibration(msight_path, intrinsics_bytes, locmap_bytes)
    return JSONResponse({
        "status": "ok",
        "message": "Calibration files uploaded and applied — they'll be used the next time the pipeline starts.",
        **calibration_status(msight_path),
    })


@router.post("/upload_video")
async def upload_video(video: UploadFile = File(...)):
    """Stream a browser-uploaded video to disk for use as video_input, enforcing the size cap."""
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        return _error(f"Unsupported file type '{suffix or '(none)'}' — expected one of {sorted(VIDEO_EXTENSIONS)}.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / f"upload_{datetime.now().strftime('%Y%m%dT%H%M%S%f')}{suffix}"
    written = 0
    try:
        with open(dest, "wb") as f:
            while chunk := await video.read(_UPLOAD_CHUNK_SIZE):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    dest.unlink(missing_ok=True)
                    return _error(f"Upload exceeds the {MAX_UPLOAD_BYTES / 1024**3:.1f}GB limit for this sandbox.", 413)
                f.write(chunk)
    except Exception as e:
        dest.unlink(missing_ok=True)
        return _error(f"Upload failed: {e}", 500)

    return JSONResponse({
        "status": "ok",
        "message": f"Uploaded '{video.filename}' ({written / 1024**2:.1f} MB).",
        "video_input": str(dest.resolve()),  # absolute: the pipeline resolves it from its own cwd
    })


@router.get("/download_recording/{filename}")
async def download_recording(filename: str):
    """Serve a finished recording as a browser download (bare filenames only, to block path traversal)."""
    if "/" in filename or "\\" in filename or filename in (".", ".."):
        return _error("Invalid filename.")
    path = (DOWNLOAD_DIR / filename).resolve()
    if DOWNLOAD_DIR.resolve() not in path.parents or not path.is_file():
        return _error("Recording not found.", 404)
    return FileResponse(path, media_type="video/mp4", filename=filename)


@router.get("/record_status")
async def record_status():
    """Live recording-progress poll for the UI's status badge, independent of the chat/LLM loop."""
    mp = WorkflowState.load().msight_pipeline
    idle = {"status": "ok", "recording_active": False, "recording_pending": False,
            "segment_count": 0, "seconds_since_last": None}
    if mp is None or not (mp.recording_active or mp.recording_pending):
        return JSONResponse({**idle, "message": "Not recording."})
    if mp.recording_pending:
        return JSONResponse({**idle, "recording_pending": True,
                             "message": "Recording will start automatically once the pipeline is running."})

    msight_path, err = get_msight_path()
    sensor = mp.sensor_name or (DEFAULT_SENSOR_NAME if err else active_sensor_name(msight_path))
    seg = recording_segment_status(sensor)
    if seg["segment_count"] == 0:
        message = "Recording — capturing first clip, nothing saved yet."
    else:
        plural = "s" if seg["segment_count"] != 1 else ""
        message = (f"Recording — {seg['segment_count']} clip{plural} captured "
                   f"(most recent {seg['seconds_since_last']}s ago). Stop recording to download.")
    return JSONResponse({"status": "ok", "recording_active": True, "recording_pending": False, **seg,
                         "message": message})


# --- live incident feed (written by the sentinel) ---------------------------------------

_FEED_BLOCK_MS = 5000  # also the keepalive interval
# A hung Redis (or a proxy in front of one) never closes the socket; this bounds how long
# a blocked read can hide an outage from the notice below.
_FEED_SOCKET_TIMEOUT_S = _FEED_BLOCK_MS / 1000 + 5
# uvicorn's graceful shutdown waits for open responses, and an endless stream would keep
# chat_server alive (not listening, still running) after Ctrl+C. Each connection ends after
# this; the browser reconnects within a second and resumes from its last event id.
_FEED_CONNECTION_SECONDS = 30


def _stream_id(value) -> tuple[int, int]:
    ms, _, seq = (value.decode() if isinstance(value, bytes) else str(value)).partition("-")
    return int(ms or 0), int(seq or 0)


def _sse(event: str, data: dict, event_id: str = "") -> str:
    return (f"id: {event_id}\n" if event_id else "") + f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _incident_sse(state: dict, event_id: str) -> str:
    return _sse("incident", {**state, "fix": incidents.fix_action(state)}, event_id)


def feed_start(recent: list[tuple[str, dict]], last_event_id: str = "") -> tuple[dict[str, dict], list[dict], str]:
    """(folded states, incidents to send now, cursor to read from) for a new connection.

    A fresh page gets every open incident. A reconnect (the browser sends the last id
    it saw) gets nothing up front: it resumes from that id, and states are folded only
    up to it so resumed events aren't applied twice."""
    if last_event_id:
        upto = _stream_id(last_event_id)
        states = incidents.fold(e for sid, e in recent if _stream_id(sid) <= upto)
        return states, [], last_event_id
    states = incidents.fold(e for _, e in recent)
    opened = sorted((s for s in states.values() if s["status"] == "open"), key=lambda s: s["detected_at"])
    return states, opened, recent[-1][0] if recent else "0-0"


_REDIS_NOTICE_AFTER_SECONDS = 3


def sentinel_offline_notice(last_seen: float) -> dict:
    """The chat server's own card when the monitor itself stopped without a clean shutdown."""
    return {"id": f"FEED-sentinel-{int(last_seen)}", "node": "sentinel", "kind": "SERVICE_DOWN", "status": "open",
            "stage": "classified", "detected_at": last_seen, "updated_at": time.time(), "steps": [],
            "summary": f"The monitor (sentinel) stopped responding at {datetime.fromtimestamp(last_seen):%H:%M:%S}. "
                       "Until it is back, failures are not detected or reported here.",
            "note": "Restart the sentinel (python -m msight_agent.sentinel)."}


def redis_outage_notice(down_since: float) -> dict:
    """The chat server's own incident card for a Redis outage: the feed itself lives in
    Redis, so the sentinel can't deliver this one (it holds its events until Redis is back)."""
    return {"id": f"FEED-redis-{int(down_since)}", "node": "redis", "kind": "SERVICE_DOWN", "status": "open",
            "stage": "classified", "detected_at": down_since, "updated_at": time.time(), "steps": [],
            "summary": "The chat server can't reach Redis, the message broker that the pipeline's data and these "
                       "monitor alerts both go through.",
            "note": "Live updates are paused; the monitor's findings will appear here once Redis is back."}


def notices_sse(active_ids: list[str]) -> str:
    return f"event: notices\ndata: {json.dumps(active_ids)}\n\n"


def _resolved_sse(notice: dict, text: str) -> str:
    return _incident_sse(incidents.apply(notice, {"incident": notice["id"], "stage": "resolved",
                                                  "at": time.time(), "text": text}), "")


def track_redis_outage(down_since: Optional[float], notice: Optional[dict]
                       ) -> tuple[float, Optional[dict], str]:
    """After one failed read: when Redis went down, its card (shown once it has been down
    a few seconds, so a blip shows nothing), and what to send ("" if nothing changed)."""
    down_since = down_since or time.time()
    if notice is None and time.time() - down_since >= _REDIS_NOTICE_AFTER_SECONDS:
        notice = redis_outage_notice(down_since)
        return down_since, notice, _incident_sse(notice, "")
    return down_since, notice, ""


def track_monitor(offline_since: Optional[float], notice: Optional[dict]) -> tuple[Optional[dict], str]:
    """The sentinel-offline card after one check, and what to send for it ("" if nothing changed)."""
    if offline_since is not None and notice is None:
        notice = sentinel_offline_notice(offline_since)
        return notice, _incident_sse(notice, "")
    if offline_since is None and notice is not None:
        return None, _resolved_sse(notice, "The monitor is running again.")
    return notice, ""


def connected_sse(opened: list[dict], cursor: str, monitor_notice: Optional[dict]) -> str:
    """A fresh connection's start: the open incidents, then which notice cards still stand.
    Notices belong to no stream entry, and one may have cleared while the browser was
    between connections."""
    active = [monitor_notice["id"]] if monitor_notice else []
    return "".join(_incident_sse(state, cursor) for state in opened) + notices_sse(active)


def new_events_sse(entries: list, states: dict[str, dict], cursor: str) -> tuple[str, str]:
    """Fold newly read stream entries into `states`; (the events to send, the new cursor)."""
    out = []
    for stream_id, fields in entries:
        cursor = stream_id.decode() if isinstance(stream_id, bytes) else stream_id
        event = incidents.decode(fields)
        if event and event.get("incident"):
            state = states[event["incident"]] = incidents.apply(states.get(event["incident"]), event)
            if state["status"] != "unknown":    # a late event of an incident from before this page
                out.append(_incident_sse(state, cursor))
    return "".join(out), cursor


# Every open chat tab holds a blocking read; they share one client (its pool) and their own
# threads, so many tabs can't use up the default executor the chat's other work runs on.
_FEED_EXECUTOR = ThreadPoolExecutor(max_workers=16, thread_name_prefix="incident-feed")
_feed_client: Optional[redis.Redis] = None


def _feed_redis() -> redis.Redis:
    global _feed_client
    if _feed_client is None:
        _feed_client = redis.Redis(**redis_kwargs(), socket_timeout=_FEED_SOCKET_TIMEOUT_S, socket_connect_timeout=3)
    return _feed_client


async def _feed_call(fn, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(_FEED_EXECUTOR, partial(fn, *args, **kwargs))


async def _offline_since(client: redis.Redis) -> Optional[float]:
    try:
        return await _feed_call(incidents.sentinel_offline_since, client)
    except redis.RedisError:
        return None   # a Redis outage has its own notice


@router.get("/incidents/stream")
async def incidents_stream(request: Request):
    """Server-sent events: each event is one incident's full current state."""
    client = _feed_redis()

    async def generate():
        states, opened, cursor = {}, [], None    # cursor stays None until the first successful read
        redis_down_since, redis_notice = None, None
        monitor_notice = None                    # the sentinel-offline card, while one is shown
        ends_at = time.monotonic() + _FEED_CONNECTION_SECONDS
        yield "retry: 1000\n\n"                # reconnect after 1s (default ~3s)
        while time.monotonic() < ends_at and not await request.is_disconnected():
            try:
                if cursor is None:
                    recent = await _feed_call(incidents.read_recent, client)
                    states, opened, cursor = feed_start(recent, request.headers.get("last-event-id", ""))
                    result = None
                else:
                    result = await _feed_call(client.xread, {incidents.STREAM_KEY: cursor},
                                              count=100, block=_FEED_BLOCK_MS)
            except redis.RedisError:
                redis_down_since, redis_notice, update = track_redis_outage(redis_down_since, redis_notice)
                if update:
                    yield update
                await asyncio.sleep(3)
                continue
            redis_down_since = None
            if redis_notice is not None:
                yield _resolved_sse(redis_notice, "Redis is reachable again; live updates have resumed.")
                redis_notice = None
            monitor_notice, update = track_monitor(await _offline_since(client), monitor_notice)
            if update:
                yield update
            if result is None:  # just (re)connected
                yield connected_sse(opened, cursor, monitor_notice)
            elif not result:
                yield ": keepalive\n\n"
            else:
                chunk, cursor = new_events_sse(result[0][1], states, cursor)
                yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
