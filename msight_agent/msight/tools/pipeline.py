"""MCP tools for the fixed demo pipeline (video source -> RF-DETR -> viewer) and
for observing whatever is running: status, diagnosis, logs."""
import asyncio
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values
from fastmcp import Context

from msight_agent.shared.host_utils import resolve_host
from msight_agent.shared.mcp import error_json, mcp, ok_json
from msight_agent.msight.runtime.calibration import calibration_mismatch
from msight_agent.msight.runtime.control_plane import get_control_plane
from msight_agent.msight.runtime.executors import has_gpu
from msight_agent.msight.runtime.paths import get_msight_path

VIEWER_PORT = 9010
FIXED_PIPELINE_NODES = ("video_source", "rfdetr_detector", "detection_viewer")
_BUILD_TIMEOUT = 600  # image build takes ~206s locally
_MAX_LOG_CHARS = 8000

_DOUBLED_ENV_RE = re.compile(r'^\s*([A-Za-z_]\w*)\s*=\s*\1\s*=', re.MULTILINE)
_LOG_TS_RE = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+')
_FRIENDLY_ERRORS = (
    (re.compile(r'refers to undefined volume'),
     "Docker Compose reports an undefined volume — check MSight_Vision's docker-compose.yml "
     "volume definitions."),
    (re.compile(r'could not select device driver ["\']?nvidia["\']?', re.IGNORECASE),
     "GPU device driver 'nvidia' could not be selected, despite nvidia-smi reporting a GPU on "
     "this host -- the NVIDIA Container Toolkit is likely missing or misconfigured."),
)


def _friendly_error(output: str) -> Optional[str]:
    return next((msg for pattern, msg in _FRIENDLY_ERRORS if pattern.search(output)), None)


def _check_msight_env(msight_path: Path, overriding_source: bool) -> Optional[str]:
    """Problems in MSight_Vision's .env; its VIDEO_INPUT is skipped when the caller overrides the source."""
    env_path = msight_path / ".env"
    if not env_path.is_file():
        return None
    text = env_path.read_text()
    if doubled := _DOUBLED_ENV_RE.search(text):
        key = doubled.group(1)
        return (f"MSight_Vision's .env has a malformed line for '{key}' (looks like "
                f"'{key}={key}=...'). Fix that line by hand before starting the pipeline.")
    if overriding_source:
        return None
    values = dotenv_values(env_path)
    video_input = values.get("VIDEO_INPUT")
    if not values.get("RTSP_URL") and video_input and not Path(video_input).exists():
        return f"VIDEO_INPUT '{video_input}' set in MSight_Vision's .env does not exist on this host."
    return None


def _render_cpu_override() -> Path:
    """cpu_override.yml with the absolute Dockerfile.cpu path filled in.
    (MSight_Vision's own docker-compose.cpu.yml doesn't clear the GPU reservation.)"""
    here = Path(__file__).resolve().parents[1] / "runtime"
    out = Path(tempfile.gettempdir()) / "msight_cpu_override.rendered.yml"
    out.write_text((here / "cpu_override.yml").read_text()
                   .replace("{DOCKERFILE_PATH}", str(here / "Dockerfile.cpu")))
    return out


async def _compose_build(msight_path: Path, ctx: Optional[Context]) -> Optional[str]:
    """`docker compose build`, streaming output via ctx.log(). Returns an error message or None."""
    files = ["-f", "docker-compose.yml"]
    if not await has_gpu():
        files += ["-f", str(_render_cpu_override())]
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "compose", *files, "--env-file", ".env", "build", cwd=str(msight_path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        return "Docker is not installed, or not on PATH, on this host."

    lines: list[str] = []

    async def stream() -> None:
        while raw := await proc.stdout.readline():
            if text := raw.decode(errors="replace").rstrip():
                lines.append(text)
                if ctx:
                    await ctx.log(text)

    try:
        await asyncio.wait_for(stream(), timeout=_BUILD_TIMEOUT)
        await proc.wait()
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return f"docker compose build timed out after {_BUILD_TIMEOUT}s."
    if proc.returncode != 0:
        output = "\n".join(lines)
        return _friendly_error(output) or f"docker compose build failed:\n{output[-2000:]}"
    return None


def _resolve_video_source(video_input: Optional[str], rtsp_url: Optional[str], sensor: str) -> tuple[str, dict]:
    """(node_type, config) for video_source, mirroring docker-compose.yml's entrypoint."""
    base = {"publish_topic": f"camera/{sensor}", "sensor_name": sensor}
    if rtsp_url:
        return "video_source_rtsp", {**base, "rtsp_url": rtsp_url}
    # Directory -> mp4_folder; a single file works with msight_launch_rtsp's --url.
    if Path(video_input).is_dir():
        return "video_source_mp4_folder", {**base, "folder": video_input}
    return "video_source_rtsp", {**base, "rtsp_url": video_input}


def _fixed_pipeline(video_node_type: str, video_config: dict, sensor: str) -> list[dict]:
    return [
        {"node_type": video_node_type, "name": "video_source", "config": video_config},
        {"node_type": "rfdetr_detector", "name": "rfdetr_detector", "config": {
            "publish_topic": f"detection/{sensor}",
            "subscribe_topic": f"camera/{sensor}",
            "det_configs": "/configs/rfdetr_config.yaml",
            "sensor_name": sensor,
        }},
        {"node_type": "detection_viewer", "name": "detection_viewer", "config": {
            "subscribe_topic": f"detection/{sensor}",
            "port": VIEWER_PORT,
        }},
    ]


def _viewer_url() -> str:
    return f"http://{resolve_host()}:{VIEWER_PORT}"


@mcp.tool()
async def start_msight_pipeline(
    video_input: Optional[str] = None,
    rtsp_url: Optional[str] = None,
    sensor_name: Optional[str] = None,
    build: bool = False,
    ctx: Context = None,
) -> str:
    msight_path, err = get_msight_path()
    if err:
        return error_json(err)
    if video_input and rtsp_url:
        return error_json("Provide exactly one of video_input or rtsp_url, not both.")
    if video_input and not Path(video_input).exists():
        return error_json(f"video_input path '{video_input}' does not exist on this host.")

    overriding_source = bool(video_input or rtsp_url)
    if not overriding_source:
        env = dotenv_values(msight_path / ".env")
        video_input, rtsp_url = env.get("VIDEO_INPUT") or None, env.get("RTSP_URL") or None
    if env_err := _check_msight_env(msight_path, overriding_source):
        return error_json(env_err)
    if not video_input and not rtsp_url:
        return error_json("No video source given, and MSight_Vision's .env sets neither VIDEO_INPUT nor RTSP_URL.")
    if mismatch := calibration_mismatch(msight_path, video_input):
        return error_json(mismatch)

    if build and (build_err := await _compose_build(msight_path, ctx)):
        return error_json(build_err)

    sensor = sensor_name or "gs_mcity_1"
    try:
        await get_control_plane().recompose(
            _fixed_pipeline(*_resolve_video_source(video_input, rtsp_url, sensor), sensor),
            progress=ctx.log if ctx else None,
        )
    except Exception as e:
        return error_json(_friendly_error(str(e)) or f"Failed to start MSight_Vision pipeline: {e}")
    return ok_json("MSight_Vision pipeline started.", viewer_url=_viewer_url())


@mcp.tool()
async def stop_msight_pipeline() -> str:
    if err := get_msight_path()[1]:
        return error_json(err)
    try:
        # Explicit names, not recompose([]), so this still works after a server restart.
        for name in FIXED_PIPELINE_NODES:
            await get_control_plane().delete_node(name)
        # Nodes added separately aren't part of the pipeline, so they're left running.
        others = await get_control_plane().live_node_names()
    except Exception as e:
        return error_json(f"Failed to stop MSight_Vision pipeline: {e}")
    if not others:
        return ok_json("MSight_Vision pipeline stopped. No other MSight nodes are running.")
    return ok_json(
        f"MSight_Vision pipeline stopped. These separately added nodes are still running: "
        f"{', '.join(others)}. Tell the user, and ask whether to remove them too.",
        other_nodes_still_running=others,
    )


@mcp.tool()
async def get_msight_status() -> str:
    if err := get_msight_path()[1]:
        return error_json(err)
    cp = get_control_plane()
    try:
        registry = cp.get_status()
        # One snapshot (one docker call) shared by liveness, problems and compute.
        snap = await cp.snapshot(n["name"] for n in registry)
    except Exception as e:
        return error_json(f"Could not read MSight_Vision status: {e}")

    now = time.time()
    services = []
    for node in registry:
        hb = node.get("last_heartbeat")
        services.append({
            **node,
            **({"seconds_since_heartbeat": max(0, int(now - hb))} if isinstance(hb, (int, float)) else {}),
            "alive": snap[node["name"]]["alive"],  # Redis status never expires
        })

    extra: dict = {"services": services}
    # Every live web viewer's page, not just the fixed pipeline's (Redis entries can be stale).
    viewer_urls = {name: f"http://{resolve_host()}:{v['web_port']}"
                   for name, v in snap.items() if v["alive"] and v.get("web_port")}
    if viewer_urls:
        extra["viewer_urls"] = viewer_urls
    if "detection_viewer" in viewer_urls:
        extra["viewer_url"] = viewer_urls["detection_viewer"]

    # Precomputed structural problems; the LLM is bad at spotting absences in a list.
    if problems := await cp.structural_check(registry, snap):
        extra["problems"] = problems
        extra["next_step"] = ("Call diagnose_msight_pipeline for the ranked root cause -- it also measures "
                              "live data flow, which catches nodes that are running but stuck.")
    elif any(s["alive"] for s in services):
        extra["data_flow"] = ("NOT MEASURED -- this only shows processes are up. If the user asked whether "
                              "the pipeline is working or healthy, call diagnose_msight_pipeline before answering.")
    extra["compute"] = await cp.compute_report({name: v["device"] for name, v in snap.items()})
    extra["recent_changes"] = cp.recent_changes()
    return ok_json(**extra)


@mcp.tool()
async def diagnose_msight_pipeline() -> str:
    if err := get_msight_path()[1]:
        return error_json(err)
    try:
        return ok_json(**await get_control_plane().diagnose())
    except Exception as e:
        return error_json(f"Could not diagnose MSight_Vision pipeline: {e}")


def _seconds_since_last_log_line(text: str) -> Optional[int]:
    """Age of msight_core's last `YYYY-MM-DD HH:MM:SS,ms` log line (a hint only)."""
    for line in reversed(text.splitlines()):
        if m := _LOG_TS_RE.match(line):
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            return max(0, int(time.time() - ts))
    return None


@mcp.tool()
async def get_msight_logs(service: Optional[str] = None, tail: int = 200) -> str:
    if err := get_msight_path()[1]:
        return error_json(err)
    tail = max(1, min(tail, 2000))
    cp = get_control_plane()
    try:
        if service:
            text = await cp.get_logs(service, tail)
            return ok_json(logs=text[-_MAX_LOG_CHARS:], seconds_since_last_line=_seconds_since_last_log_line(text))
        per_node = {n["name"]: await cp.get_logs(n["name"], tail) for n in cp.get_status()}
    except Exception as e:
        return error_json(f"Could not fetch logs: {e}")
    combined = "\n\n".join(f"==> {name} <==\n{text}" for name, text in per_node.items())
    return ok_json(
        logs=combined[-_MAX_LOG_CHARS:],
        log_freshness={name: _seconds_since_last_log_line(text) for name, text in per_node.items()},
    )
