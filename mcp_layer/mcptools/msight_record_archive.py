"""Record & Archive: nodes that record the *annotated* feed, run through the
same MSightControlPlane the main pipeline (msight_docker.py) uses -- these 4
node types are cataloged with invocation="supervisor" (never Docker, per this
file's original design point: avoid touching the MSight_Vision checkout).

    video_source --> camera/$SENSOR_NAME --> rfdetr_detector --> detection/$SENSOR_NAME
        --> annotated_frame_publisher (ours, msight_nodes/) --> annotated/$SENSOR_NAME
        --> image_to_video_aggregator (unmodified) --> video/$SENSOR_NAME --> video_local_dumper / aws_video_pusher

The annotator/aggregator dependency chain below (ensure-parent-before-child,
don't tear down a shared parent while a sibling sink still needs it) is this
file's own orchestration logic, not something the generic control plane
models -- it's preserved as-is, just delegating node start/stop through it.
"""
import asyncio
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from mcptools import mcp
from mcptools.mcp_json import error_json, ok_json
from mcptools.msight_docker import _control_plane, _get_msight_path

# Sensor the current/most-recent recording session used -- lets
# stop_msight_recording find the right segment folder without a
# sensor_name parameter stop calls don't otherwise need.
_LAST_RECORDING_SENSOR: Optional[str] = None

# Finished, single-file recordings for chat_server.py's
# /msight/download_recording route -- separate from the raw per-segment
# save_dir so a download is always one file, never a folder of chunks.
DOWNLOAD_DIR = Path("output/msight_downloads")

ANNOTATOR_NODE = "frame_annotator"
AGGREGATOR_NODE = "video_aggregator"
DUMPER_NODE = "local_dumper"
PUSHER_NODE = "s3_pusher"

DEFAULT_SENSOR_NAME = "gs_mcity_1"
# The aggregator only publishes a clip once it's collected this many
# frames -- no time-based fallback, so a shorter session silently produces
# zero segments. Kept low so a short test session still produces one; real
# deployments just get more, smaller segments, auto-concatenated on stop.
DEFAULT_BUFFER_SIZE = 40
DEFAULT_OVERLAP_SIZE = 0
DEFAULT_FPS = 20


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


async def _ensure_annotator(msight_path: Path, sensor_name: str) -> tuple[bool, str]:
    """Idempotent, same pattern as _ensure_aggregator -- draws boxes onto detections and republishes for the aggregator to consume."""
    cp = _control_plane()
    if cp.is_tracked(ANNOTATOR_NODE):
        return True, ""
    try:
        await cp.add_node("frame_annotator", name=ANNOTATOR_NODE, config={
            "subscribe_topic": f"detection/{sensor_name}",
            "publish_topic": f"annotated/{sensor_name}",
        })
        return True, ""
    except Exception as e:
        return False, f"Could not start {ANNOTATOR_NODE}: {e}"


async def _ensure_aggregator(msight_path: Path, sensor_name: str) -> tuple[bool, str]:
    """Idempotent: archiving alone still needs the aggregator running,
    since the S3 pusher subscribes to the aggregator's video/ output, not
    the annotated frame feed directly. Starting recording first is not
    required.

    Known gap: if the aggregator is already alive, sensor_name isn't
    checked against what it was actually launched with -- a topic mismatch
    from a prior call would go unnoticed here."""
    cp = _control_plane()
    if cp.is_tracked(AGGREGATOR_NODE):
        return True, ""
    ok, msg = await _ensure_annotator(msight_path, sensor_name)
    if not ok:
        return False, msg
    try:
        await cp.add_node("video_aggregator", name=AGGREGATOR_NODE, config={
            "subscribe_topic": f"annotated/{sensor_name}",
            "publish_topic": f"video/{sensor_name}",
            "buffer_size": DEFAULT_BUFFER_SIZE,
            "overlap_size": DEFAULT_OVERLAP_SIZE,
            "fps": DEFAULT_FPS,
        })
        return True, ""
    except Exception as e:
        return False, f"Could not start {AGGREGATOR_NODE}: {e}"


@mcp.tool()
async def start_msight_recording(sensor_name: Optional[str] = None) -> str:
    """Start local recording of the annotated feed: frame annotator + aggregator + local disk dumper."""
    msight_path, err = _get_msight_path()
    if err:
        return error_json(err)
    sensor = sensor_name or _active_sensor_name(msight_path)

    ok, msg = await _ensure_aggregator(msight_path, sensor)
    if not ok:
        return error_json(msg)

    save_dir = os.environ.get("MSIGHT_RECORDING_SAVE_DIR", "output/msight_recordings")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    try:
        await _control_plane().add_node("video_local_dumper", name=DUMPER_NODE, config={
            "subscribe_topic": f"video/{sensor}",
            "save_dir": str(Path(save_dir).resolve()),
        })
    except Exception as e:
        return error_json(f"Could not start {DUMPER_NODE}: {e}")
    global _LAST_RECORDING_SENSOR
    _LAST_RECORDING_SENSOR = sensor
    return ok_json(
        "Recording is set up. It writes video in chunks — nothing is saved to disk "
        f"until it has buffered {DEFAULT_BUFFER_SIZE} frames from the pipeline, so if "
        "the pipeline isn't running yet, or you stop again within a few seconds, there "
        "may be nothing to save yet. stop_msight_recording will combine whatever "
        "chunks did get written into one downloadable file."
    )


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
        return error_json(err)
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return error_json(
            "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are not set in .env — "
            "archiving needs AWS credentials to write to S3."
        )
    sensor = _active_sensor_name(msight_path)

    ok, msg = await _ensure_aggregator(msight_path, sensor)
    if not ok:
        return error_json(msg)

    try:
        await _control_plane().add_node("aws_video_pusher", name=PUSHER_NODE, config={
            "subscribe_topic": f"video/{sensor}",
            "bucket_name": s3_bucket,
            "prefix": s3_prefix or "",
        })
    except Exception as e:
        return error_json(f"Could not start {PUSHER_NODE}: {e}")
    return ok_json(f"Archiving started — video files will be pushed to s3://{s3_bucket}/{s3_prefix or ''}.")


async def _stop_aggregator_chain() -> str:
    """Stop the aggregator and its upstream annotator together."""
    cp = _control_plane()
    agg_was_up = cp.is_tracked(AGGREGATOR_NODE)
    ann_was_up = cp.is_tracked(ANNOTATOR_NODE)
    await cp.delete_node(AGGREGATOR_NODE)
    await cp.delete_node(ANNOTATOR_NODE)
    agg_msg = f"{AGGREGATOR_NODE} stopped." if agg_was_up else f"{AGGREGATOR_NODE} was not running."
    ann_msg = f"{ANNOTATOR_NODE} stopped." if ann_was_up else f"{ANNOTATOR_NODE} was not running."
    return f"{agg_msg} {ann_msg}"


@mcp.tool()
async def stop_msight_recording() -> str:
    """Stop the local dumper (and aggregator/annotator if archiving isn't also active), then combine segments into one downloadable .mp4."""
    cp = _control_plane()
    dumper_was_up = cp.is_tracked(DUMPER_NODE)
    await cp.delete_node(DUMPER_NODE)
    msg = f"{DUMPER_NODE} stopped." if dumper_was_up else f"{DUMPER_NODE} was not running."
    agg_msg = ""
    if not cp.is_tracked(PUSHER_NODE):
        agg_msg = " " + await _stop_aggregator_chain()

    global _LAST_RECORDING_SENSOR
    sensor = _LAST_RECORDING_SENSOR
    _LAST_RECORDING_SENSOR = None
    if not sensor:
        return ok_json(f"{msg}{agg_msg}")

    save_dir = Path(os.environ.get("MSIGHT_RECORDING_SAVE_DIR", "output/msight_recordings"))
    out_path, err = await _concat_recording_segments(save_dir, sensor)
    if err:
        return ok_json(f"{msg}{agg_msg} {err}")

    return ok_json(f"{msg}{agg_msg} Recording saved as {out_path.name}.", download_filename=out_path.name)


@mcp.tool()
async def stop_msight_archiving() -> str:
    """Stop the S3 pusher (and aggregator/annotator if recording isn't also active)."""
    cp = _control_plane()
    pusher_was_up = cp.is_tracked(PUSHER_NODE)
    await cp.delete_node(PUSHER_NODE)
    msg = f"{PUSHER_NODE} stopped." if pusher_was_up else f"{PUSHER_NODE} was not running."
    if not cp.is_tracked(DUMPER_NODE):
        agg_msg = await _stop_aggregator_chain()
        return ok_json(f"{msg} {agg_msg}")
    return ok_json(msg)


@mcp.tool()
async def get_msight_record_archive_status() -> str:
    """Report which record/archive nodes are actually alive right now.
    Uses is_alive(), not is_tracked() -- presence in memory isn't liveness."""
    cp = _control_plane()
    return ok_json(
        frame_annotator=await cp.is_alive(ANNOTATOR_NODE),
        video_aggregator=await cp.is_alive(AGGREGATOR_NODE),
        local_dumper=await cp.is_alive(DUMPER_NODE),
        s3_pusher=await cp.is_alive(PUSHER_NODE),
    )
