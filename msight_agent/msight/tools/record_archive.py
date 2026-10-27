"""Record & Archive: supervisor-run nodes that record the annotated feed.

    video_source --> camera/$SENSOR_NAME --> rfdetr_detector --> detection/$SENSOR_NAME
        --> annotated_frame_publisher (upstream/msight_vision_nodes/) --> annotated/$SENSOR_NAME
        --> image_to_video_aggregator --> video/$SENSOR_NAME --> video_local_dumper / aws_video_pusher

The annotator/aggregator are shared parents: started before either sink and
kept alive while any sink still needs them.
"""
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values

from msight_agent.shared.mcp import error_json, mcp, ok_json
from msight_agent.msight.runtime.control_plane import get_control_plane
from msight_agent.msight.runtime.paths import get_msight_path
from msight_agent.msight.runtime.utils import run_cmd
from msight_agent.shared.paths import MSIGHT_DOWNLOADS_DIR, MSIGHT_RECORDINGS_DIR

# Concatenated single-file recordings served by /msight/download_recording.
DOWNLOAD_DIR = MSIGHT_DOWNLOADS_DIR

ANNOTATOR_NODE = "frame_annotator"
AGGREGATOR_NODE = "video_aggregator"
DUMPER_NODE = "local_dumper"
PUSHER_NODE = "s3_pusher"

DEFAULT_SENSOR_NAME = "gs_mcity_1"
# Frames per aggregator clip; kept low since shorter sessions produce no clip at all.
DEFAULT_BUFFER_SIZE = 40
DEFAULT_OVERLAP_SIZE = 0
DEFAULT_FPS = 20

# Sensor of the latest recording session, so stop can find its segments.
_last_recording_sensor: Optional[str] = None


def _recording_dir() -> Path:
    return MSIGHT_RECORDINGS_DIR


def _segments(sensor: str) -> list[Path]:
    """This sensor's recorded clips; name order is chronological."""
    segment_dir = _recording_dir() / sensor
    return sorted(segment_dir.glob(f"{sensor}_*.mp4")) if segment_dir.is_dir() else []


def active_sensor_name(msight_path: Path) -> str:
    """Sensor name the running pipeline uses (from MSight_Vision's .env).
    A wrong guess leaves the aggregator silently recording nothing."""
    return dotenv_values(msight_path / ".env").get("SENSOR_NAME") or DEFAULT_SENSOR_NAME


def recording_segment_status(sensor: str) -> dict:
    """Live on-disk scan of recorded segments."""
    segments = _segments(sensor)
    if not segments:
        return {"segment_count": 0, "seconds_since_last": None}
    last_mtime = max(p.stat().st_mtime for p in segments)
    return {"segment_count": len(segments), "seconds_since_last": max(0, int(time.time() - last_mtime))}


async def _ensure_node(name: str, node_type: str, config: dict) -> Optional[str]:
    """Start `name` unless it's already alive (a pidfile-recovered process counts, so
    it isn't duplicated after a server restart). Returns an error message or None."""
    if await get_control_plane().is_alive(name):
        return None
    try:
        await get_control_plane().add_node(node_type, name=name, config=config)
    except Exception as e:
        return f"Could not start {name}: {e}"
    return None


async def _ensure_aggregator(sensor: str) -> Optional[str]:
    """Annotator + aggregator, shared by recording and archiving.
    Known gap: an already-running aggregator isn't checked for a sensor mismatch."""
    return (
        await _ensure_node(ANNOTATOR_NODE, "frame_annotator", {
            "subscribe_topic": f"detection/{sensor}",
            "publish_topic": f"annotated/{sensor}",
        })
        or await _ensure_node(AGGREGATOR_NODE, "video_aggregator", {
            "subscribe_topic": f"annotated/{sensor}",
            "publish_topic": f"video/{sensor}",
            "buffer_size": DEFAULT_BUFFER_SIZE,
            "overlap_size": DEFAULT_OVERLAP_SIZE,
            "fps": DEFAULT_FPS,
        })
    )


async def _stop_node(name: str) -> str:
    was_up = get_control_plane().is_tracked(name)
    await get_control_plane().delete_node(name)
    return f"{name} stopped." if was_up else f"{name} was not running."


async def _stop_aggregator_chain() -> str:
    return f"{await _stop_node(AGGREGATOR_NODE)} {await _stop_node(ANNOTATOR_NODE)}"


@mcp.tool()
async def start_msight_recording(sensor_name: Optional[str] = None) -> str:
    """Start local recording of the annotated feed: frame annotator + aggregator + local disk dumper."""
    global _last_recording_sensor
    msight_path, err = get_msight_path()
    if err:
        return error_json(err)
    sensor = sensor_name or active_sensor_name(msight_path)
    save_dir = _recording_dir()
    save_dir.mkdir(parents=True, exist_ok=True)
    if err := await _ensure_aggregator(sensor) or await _ensure_node(DUMPER_NODE, "video_local_dumper", {
        "subscribe_topic": f"video/{sensor}",
        "save_dir": str(save_dir.resolve()),
    }):
        return error_json(err)
    _last_recording_sensor = sensor
    return ok_json(
        "Recording is set up. It writes video in chunks — nothing is saved to disk "
        f"until it has buffered {DEFAULT_BUFFER_SIZE} frames from the pipeline, so if "
        "the pipeline isn't running yet, or you stop again within a few seconds, there "
        "may be nothing to save yet. stop_msight_recording will combine whatever "
        "chunks did get written into one downloadable file."
    )


@mcp.tool()
async def start_msight_archiving(s3_bucket: str, s3_prefix: Optional[str] = None) -> str:
    """Start S3 archiving of the annotated feed: frame annotator + aggregator + S3 pusher. Needs AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY."""
    msight_path, err = get_msight_path()
    if err:
        return error_json(err)
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return error_json("AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are not set in .env — "
                          "archiving needs AWS credentials to write to S3.")
    sensor = active_sensor_name(msight_path)
    if err := await _ensure_aggregator(sensor) or await _ensure_node(PUSHER_NODE, "aws_video_pusher", {
        "subscribe_topic": f"video/{sensor}",
        "bucket_name": s3_bucket,
        "prefix": s3_prefix or "",
    }):
        return error_json(err)
    return ok_json(f"Archiving started — video files will be pushed to s3://{s3_bucket}/{s3_prefix or ''}.")


async def _concat_recording_segments(sensor: str) -> tuple[Optional[Path], Optional[str]]:
    """Concat this session's segments into one .mp4 with ffmpeg, then delete them."""
    segments = _segments(sensor)
    if not segments:
        return None, (
            f"No recorded video segments were found — recording only writes a chunk "
            f"once it has buffered {DEFAULT_BUFFER_SIZE} frames from the pipeline (no "
            f"partial chunk is saved), so the session was likely stopped before that "
            f"happened. This isn't an error — try leaving the recording running longer, "
            f"or confirm the pipeline was actually running and producing frames the "
            f"whole time recording was on."
        )
    segment_dir = segments[0].parent
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None, ("ffmpeg is not installed on this host — the recorded segments are still on "
                      f"disk at {segment_dir}, but they could not be combined into one file.")

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOWNLOAD_DIR / f"{sensor}_recording_{datetime.now().strftime('%Y%m%dT%H%M%S')}.mp4"
    list_path = segment_dir / "_concat_list.txt"
    list_path.write_text("\n".join(f"file '{p.resolve()}'" for p in segments))
    code, out = await run_cmd(ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
                              "-c", "copy", str(out_path))
    list_path.unlink(missing_ok=True)
    if code != 0 or not out_path.is_file():
        return None, f"ffmpeg failed to combine the recorded segments:\n{out[-1000:]}"

    for p in segments:
        p.unlink(missing_ok=True)
        p.with_name(f"{p.stem}_metadata.json").unlink(missing_ok=True)
    return out_path, None


@mcp.tool()
async def stop_msight_recording() -> str:
    """Stop the local dumper (and aggregator/annotator if archiving isn't also active), then combine segments into one downloadable .mp4."""
    global _last_recording_sensor
    msg = await _stop_node(DUMPER_NODE)
    if not get_control_plane().is_tracked(PUSHER_NODE):
        msg += " " + await _stop_aggregator_chain()

    sensor, _last_recording_sensor = _last_recording_sensor, None
    if not sensor:
        # Lost on an MCP server restart mid-recording; the pipeline's sensor is the
        # only one recordings are made for, so use it if it has segments on disk.
        msight_path, err = get_msight_path()
        fallback = None if err else active_sensor_name(msight_path)
        if not fallback or not _segments(fallback):
            return ok_json(msg)
        sensor = fallback
    out_path, err = await _concat_recording_segments(sensor)
    if err:
        return ok_json(f"{msg} {err}")
    return ok_json(f"{msg} Recording saved as {out_path.name}.", download_filename=out_path.name)


@mcp.tool()
async def stop_msight_archiving() -> str:
    """Stop the S3 pusher (and aggregator/annotator if recording isn't also active)."""
    msg = await _stop_node(PUSHER_NODE)
    if not get_control_plane().is_tracked(DUMPER_NODE):
        msg += " " + await _stop_aggregator_chain()
    return ok_json(msg)


@mcp.tool()
async def get_msight_record_archive_status() -> str:
    """Which record/archive nodes are actually alive right now (liveness, not tracking)."""
    snap = await get_control_plane().snapshot(
        [ANNOTATOR_NODE, AGGREGATOR_NODE, DUMPER_NODE, PUSHER_NODE])
    return ok_json(
        frame_annotator=snap[ANNOTATOR_NODE]["alive"],
        video_aggregator=snap[AGGREGATOR_NODE]["alive"],
        local_dumper=snap[DUMPER_NODE]["alive"],
        s3_pusher=snap[PUSHER_NODE]["alive"],
    )
