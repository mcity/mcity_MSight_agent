"""MSight file endpoints served by chat_server.py for the chat UI: calibration and
video uploads, recording download, and the recording-progress badge."""
import io
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from msight_agent.msight.runtime.calibration import calibration_status, locmap_problem, write_calibration
from msight_agent.msight.runtime.paths import get_msight_path
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
