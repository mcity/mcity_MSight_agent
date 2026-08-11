"""Record & Archive: tracked host subprocesses (not Docker containers) that record the *annotated* feed.

    video_source --> camera/$SENSOR_NAME --> rfdetr_detector --> detection/$SENSOR_NAME
        --> annotated_frame_publisher (ours, msight_nodes/) --> annotated/$SENSOR_NAME
        --> image_to_video_aggregator (unmodified) --> video/$SENSOR_NAME --> video_local_dumper / aws_video_pusher

Host subprocesses rather than a Docker Compose override, to avoid touching the MSight_Vision checkout at all.
"""
import asyncio
import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from mcptools import mcp
from mcptools.msight_docker import _get_msight_path

# In-memory only, single-flight (like progress_relay._active_progress_cb) --
# lost on mcp_server.py restart, and a second concurrent session would
# overwrite the first's tracked handle. Matches this app's existing
# single-session assumption.
_ACTIVE: dict[str, asyncio.subprocess.Process] = {}

# Sensor the current/most-recent recording session used -- lets
# stop_msight_recording find the right segment folder without a
# sensor_name parameter stop calls don't otherwise need.
_LAST_RECORDING_SENSOR: Optional[str] = None

LOG_DIR = Path("output/logs/msight_record_archive")

# Finished, single-file recordings for chat_server.py's
# /msight/download_recording route -- separate from the raw per-segment
# save_dir so a download is always one file, never a folder of chunks.
DOWNLOAD_DIR = Path("output/msight_downloads")

ANNOTATOR_NODE = "frame_annotator"
AGGREGATOR_NODE = "video_aggregator"
DUMPER_NODE = "local_dumper"
PUSHER_NODE = "s3_pusher"

# Ours, launched with MSight_Vision's own venv interpreter.
_ANNOTATOR_SCRIPT = (
    Path(__file__).resolve().parents[1] / "msight_nodes" / "annotated_frame_publisher.py"
)

DEFAULT_SENSOR_NAME = "gs_mcity_1"
# The aggregator only publishes a clip once it's collected this many
# frames -- no time-based fallback, so a shorter session silently produces
# zero segments. Kept low so a short test session still produces one; real
# deployments just get more, smaller segments, auto-concatenated on stop.
DEFAULT_BUFFER_SIZE = 40
DEFAULT_OVERLAP_SIZE = 0
DEFAULT_FPS = 20


def _venv_bin(msight_path: Path, name: str) -> Optional[Path]:
    candidate = msight_path / "venv" / "bin" / name
    return candidate if candidate.is_file() else None


def _active_sensor_name(msight_path: Path) -> str:
    """Resolve the sensor_name the *running* pipeline actually uses, by
    reading MSight_Vision's own .env directly rather than guessing --
    video_source falls back to .env's SENSOR_NAME whenever
    start_msight_pipeline doesn't override it, and a mismatched guess here
    would leave the aggregator subscribed to a topic nothing publishes to
    (stays alive, looks fine, silently records nothing)."""
    env_path = msight_path / ".env"
    if env_path.is_file():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("SENSOR_NAME=") and not line.startswith("#"):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    return value
    return DEFAULT_SENSOR_NAME


def recording_segment_status(sensor: str) -> dict:
    """Live on-disk scan of recorded segments -- the only honest way to know if a clip has landed yet."""
    save_dir = Path(os.environ.get("MSIGHT_RECORDING_SAVE_DIR", "output/msight_recordings"))
    segment_dir = save_dir / sensor
    segments = sorted(segment_dir.glob(f"{sensor}_*.mp4")) if segment_dir.is_dir() else []
    if not segments:
        return {"segment_count": 0, "seconds_since_last": None}
    last_mtime = max(p.stat().st_mtime for p in segments)
    return {
        "segment_count": len(segments),
        "seconds_since_last": max(0, int(time.time() - last_mtime)),
    }


def _is_alive(name: str) -> bool:
    proc = _ACTIVE.get(name)
    return proc is not None and proc.returncode is None


# Wait this long after spawning before trusting the process is actually up
# -- msight_core nodes fail fast (missing binary, bad Redis connection), so
# this catches an immediate crash instead of declaring success the instant
# the OS hands back a PID.
_STARTUP_GRACE_SECONDS = 0.5


async def _launch(name: str, cmd: list[str], msight_path: Path) -> tuple[bool, str]:
    """Spawn a detached background node, redirecting output to its own log
    file (no live streaming -- unlike docker compose's --build, these start
    near-instantly, so there's nothing worth streaming)."""
    if _is_alive(name):
        return True, f"{name} is already running."

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    log_file = open(log_path, "wb")

    env = os.environ.copy()
    env.setdefault("MSIGHT_EDGE_DEVICE_NAME", "mcity_edge")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=log_file, stderr=asyncio.subprocess.STDOUT,
            env=env, cwd=str(msight_path), start_new_session=True,
        )
    except FileNotFoundError as e:
        log_file.close()
        return False, f"Could not launch {name}: {e}"
    finally:
        log_file.close()

    await asyncio.sleep(_STARTUP_GRACE_SECONDS)
    if proc.returncode is not None:
        tail = log_path.read_text(errors="replace")[-1000:] if log_path.is_file() else ""
        logging.warning(
            f"[RECORD_ARCHIVE] {name} exited immediately (code {proc.returncode}): {tail}"
        )
        return False, (
            f"{name} started but exited immediately (exit code {proc.returncode}). "
            f"Last output:\n{tail}" if tail else
            f"{name} started but exited immediately (exit code {proc.returncode})."
        )

    _ACTIVE[name] = proc
    logging.warning(f"[RECORD_ARCHIVE] Started {name} (pid {proc.pid}), logging to {log_path}")
    return True, f"{name} started (pid {proc.pid})."


async def _stop(name: str) -> str:
    proc = _ACTIVE.get(name)
    if proc is None or proc.returncode is not None:
        _ACTIVE.pop(name, None)
        return f"{name} was not running."
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
    _ACTIVE.pop(name, None)
    return f"{name} stopped."


async def _ensure_annotator(msight_path: Path, sensor_name: str) -> tuple[bool, str]:
    """Idempotent, same pattern as _ensure_aggregator -- draws boxes onto detections and republishes for the aggregator to consume."""
    if _is_alive(ANNOTATOR_NODE):
        return True, ""
    python_bin = _venv_bin(msight_path, "python3")
    if python_bin is None:
        return False, "python3 not found in MSight_Vision's venv."
    cmd = [
        str(python_bin), str(_ANNOTATOR_SCRIPT),
        "--name", ANNOTATOR_NODE,
        "--subscribe-topic", f"detection/{sensor_name}",
        "--publish-topic", f"annotated/{sensor_name}",
    ]
    return await _launch(ANNOTATOR_NODE, cmd, msight_path)


async def _ensure_aggregator(msight_path: Path, sensor_name: str) -> tuple[bool, str]:
    """Idempotent: archiving alone still needs the aggregator running,
    since the S3 pusher subscribes to the aggregator's video/ output, not
    the annotated frame feed directly. Starting recording first is not
    required.

    Known gap: if the aggregator is already alive, sensor_name isn't
    checked against what it was actually launched with -- a topic mismatch
    from a prior call would go unnoticed here."""
    if _is_alive(AGGREGATOR_NODE):
        return True, ""
    ok, msg = await _ensure_annotator(msight_path, sensor_name)
    if not ok:
        return False, msg
    binary = _venv_bin(msight_path, "msight_launch_image_to_video_aggregator")
    if binary is None:
        return False, "msight_launch_image_to_video_aggregator not found in MSight_Vision's venv."
    cmd = [
        str(binary),
        "--name", AGGREGATOR_NODE,
        "--subscribe-topic", f"annotated/{sensor_name}",
        "--publish-topic", f"video/{sensor_name}",
        "--buffer-size", str(DEFAULT_BUFFER_SIZE),
        "--overlap-size", str(DEFAULT_OVERLAP_SIZE),
        "--fps", str(DEFAULT_FPS),
    ]
    return await _launch(AGGREGATOR_NODE, cmd, msight_path)


@mcp.tool()
async def start_msight_recording(sensor_name: Optional[str] = None) -> str:
    """Start local recording of the annotated feed: frame annotator + aggregator + local disk dumper."""
    msight_path, err = _get_msight_path()
    if err:
        return json.dumps({"status": "error", "message": err})
    sensor = sensor_name or _active_sensor_name(msight_path)

    ok, msg = await _ensure_aggregator(msight_path, sensor)
    if not ok:
        return json.dumps({"status": "error", "message": msg})

    binary = _venv_bin(msight_path, "msight_launch_video_local_dumper")
    if binary is None:
        return json.dumps({
            "status": "error",
            "message": "msight_launch_video_local_dumper not found in MSight_Vision's venv.",
        })
    save_dir = os.environ.get("MSIGHT_RECORDING_SAVE_DIR", "output/msight_recordings")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary),
        "--name", DUMPER_NODE,
        "--subscribe-topic", f"video/{sensor}",
        "--save-dir", str(Path(save_dir).resolve()),
    ]
    ok, msg = await _launch(DUMPER_NODE, cmd, msight_path)
    if not ok:
        return json.dumps({"status": "error", "message": msg})
    global _LAST_RECORDING_SENSOR
    _LAST_RECORDING_SENSOR = sensor
    return json.dumps({
        "status": "ok",
        "message": (
            "Recording is set up. It writes video in chunks — nothing is saved to disk "
            f"until it has buffered {DEFAULT_BUFFER_SIZE} frames from the pipeline, so if "
            "the pipeline isn't running yet, or you stop again within a few seconds, there "
            "may be nothing to save yet. stop_msight_recording will combine whatever "
            "chunks did get written into one downloadable file."
        ),
    })


async def _concat_recording_segments(save_dir: Path, sensor: str) -> tuple[Optional[Path], Optional[str]]:
    """Combine this session's .mp4 segments (one per aggregator buffer, named
    "<sensor>_<capture_timestamp>.mp4" -- lexicographic sort is chronological)
    into one file via ffmpeg's concat demuxer. Stream copy is safe since
    every segment shares an encoder/pipeline run. Segments are deleted after
    a successful concat so a later session's segments don't get merged in."""
    segment_dir = save_dir / sensor
    segments = sorted(segment_dir.glob(f"{sensor}_*.mp4"))
    if not segments:
        return None, (
            f"No recorded video segments were found — recording only writes a chunk "
            f"once it has buffered {DEFAULT_BUFFER_SIZE} frames from the pipeline (no "
            f"partial chunk is saved), so the session was likely stopped before that "
            f"happened. This isn't an error — try leaving the recording running longer, "
            f"or confirm the pipeline was actually running and producing frames the "
            f"whole time recording was on."
        )

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None, (
            "ffmpeg is not installed on this host — the recorded segments are still on "
            f"disk at {segment_dir}, but they could not be combined into one file."
        )

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = DOWNLOAD_DIR / f"{sensor}_recording_{timestamp}.mp4"

    list_path = segment_dir / "_concat_list.txt"
    list_path.write_text("\n".join(f"file '{p.resolve()}'" for p in segments))

    proc = await asyncio.create_subprocess_exec(
        ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-c", "copy", str(out_path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    list_path.unlink(missing_ok=True)

    if proc.returncode != 0 or not out_path.is_file():
        tail = stdout.decode(errors="replace")[-1000:] if stdout else ""
        return None, f"ffmpeg failed to combine the recorded segments:\n{tail}"

    for p in segments:
        p.unlink(missing_ok=True)
        p.with_name(f"{p.stem}_metadata.json").unlink(missing_ok=True)

    return out_path, None


@mcp.tool()
async def start_msight_archiving(s3_bucket: str, s3_prefix: Optional[str] = None) -> str:
    """Start S3 archiving of the annotated feed: frame annotator + aggregator + S3 pusher. Needs AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY."""
    msight_path, err = _get_msight_path()
    if err:
        return json.dumps({"status": "error", "message": err})
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return json.dumps({
            "status": "error",
            "message": "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are not set in .env — "
                       "archiving needs AWS credentials to write to S3.",
        })
    sensor = _active_sensor_name(msight_path)

    ok, msg = await _ensure_aggregator(msight_path, sensor)
    if not ok:
        return json.dumps({"status": "error", "message": msg})

    binary = _venv_bin(msight_path, "msight_launch_aws_video_pusher")
    if binary is None:
        return json.dumps({
            "status": "error",
            "message": "msight_launch_aws_video_pusher not found in MSight_Vision's venv.",
        })
    cmd = [
        str(binary),
        "--name", PUSHER_NODE,
        "--subscribe-topic", f"video/{sensor}",
        "--bucket-name", s3_bucket,
        "--prefix", s3_prefix or "",
    ]
    ok, msg = await _launch(PUSHER_NODE, cmd, msight_path)
    if not ok:
        return json.dumps({"status": "error", "message": msg})
    return json.dumps({
        "status": "ok",
        "message": f"Archiving started — video files will be pushed to s3://{s3_bucket}/{s3_prefix or ''}.",
    })


async def _stop_aggregator_chain() -> str:
    """Stop the aggregator and its upstream annotator together."""
    agg_msg = await _stop(AGGREGATOR_NODE)
    ann_msg = await _stop(ANNOTATOR_NODE)
    return f"{agg_msg} {ann_msg}"


@mcp.tool()
async def stop_msight_recording() -> str:
    """Stop the local dumper (and aggregator/annotator if archiving isn't also active), then combine segments into one downloadable .mp4."""
    msg = await _stop(DUMPER_NODE)
    agg_msg = ""
    if not _is_alive(PUSHER_NODE):
        agg_msg = " " + await _stop_aggregator_chain()

    global _LAST_RECORDING_SENSOR
    sensor = _LAST_RECORDING_SENSOR
    _LAST_RECORDING_SENSOR = None
    if not sensor:
        return json.dumps({"status": "ok", "message": f"{msg}{agg_msg}"})

    save_dir = Path(os.environ.get("MSIGHT_RECORDING_SAVE_DIR", "output/msight_recordings"))
    out_path, err = await _concat_recording_segments(save_dir, sensor)
    if err:
        return json.dumps({"status": "ok", "message": f"{msg}{agg_msg} {err}"})

    return json.dumps({
        "status": "ok",
        "message": f"{msg}{agg_msg} Recording saved as {out_path.name}.",
        "download_filename": out_path.name,
    })


@mcp.tool()
async def stop_msight_archiving() -> str:
    """Stop the S3 pusher (and aggregator/annotator if recording isn't also active)."""
    msg = await _stop(PUSHER_NODE)
    if not _is_alive(DUMPER_NODE):
        agg_msg = await _stop_aggregator_chain()
        return json.dumps({"status": "ok", "message": f"{msg} {agg_msg}"})
    return json.dumps({"status": "ok", "message": msg})


@mcp.tool()
def get_msight_record_archive_status() -> str:
    """Report which record/archive nodes are currently running (this
    process's own tracking, not docker compose ps — these aren't
    containers)."""
    return json.dumps({
        "frame_annotator": _is_alive(ANNOTATOR_NODE),
        "video_aggregator": _is_alive(AGGREGATOR_NODE),
        "local_dumper": _is_alive(DUMPER_NODE),
        "s3_pusher": _is_alive(PUSHER_NODE),
    })
