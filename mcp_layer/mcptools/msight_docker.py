import os
import re
import time
import hashlib
import asyncio
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastmcp import Context

from mcptools import mcp
from mcptools.mcp_json import error_json, ok_json
from mcptools.msight_control_plane import MSightControlPlane
from mcptools.msight_executors import has_gpu
from host_utils import resolve_host

load_dotenv()

VIEWER_PORT = 9010
COMPOSE_TIMEOUT_UP = 600  # image build takes ~206s locally
FIXED_PIPELINE_NODES = ("video_source", "rfdetr_detector", "detection_viewer")

# Fixed calibration file locations, matching what rfdetr_config.yaml points at.
CALIBRATION_INTRINSICS_REL = Path("examples/rfdetr/calibration/intrinsics.json")
CALIBRATION_LOCMAP_REL = Path("examples/rfdetr/locmaps/locmap_sip_gs_Fuller_Glazier2_v1.npz")

# SHA256 of the shipped demo calibration, to tell default from user-uploaded.
_DEFAULT_INTRINSICS_SHA256 = "a04a32ac2bb4e7b54d58769b37cb79c9c7447b4b46c2279db65d05fb2c3eb57a"
_DEFAULT_LOCMAP_SHA256 = "367ca8bfc6446efb5b7e1ba3ff5704da66bdcc5d70c35b2415f597c41a2eddd7"

_DOUBLED_ENV_RE = re.compile(r'^\s*([A-Za-z_]\w*)\s*=\s*\1\s*=', re.MULTILINE)
_UNDEFINED_VOL_RE = re.compile(r'refers to undefined volume')
_NVIDIA_ERR_RE = re.compile(r'could not select device driver ["\']?nvidia["\']?', re.IGNORECASE)
_REDIS_PORT_ERR_RE = re.compile(r'Failed listening on port 6379|dependency redis failed to start', re.IGNORECASE)


def _get_msight_path() -> tuple[Optional[Path], Optional[str]]:
    load_dotenv(override=True)
    raw = os.environ.get("MSIGHT_VISION_PATH")
    if not raw:
        return None, "MSIGHT_VISION_PATH is not set in .env."
    path = Path(raw)
    if not path.is_dir():
        return None, f"MSIGHT_VISION_PATH ('{path}') does not exist or is not a directory."
    if not (path / "docker-compose.yml").is_file():
        return None, f"MSIGHT_VISION_PATH ('{path}') does not contain a docker-compose.yml."
    return path, None


def _sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _calibration_status(msight_path: Path) -> dict:
    """Live calibration check (also called directly by chat_server's state hint)."""
    intrinsics_path = msight_path / CALIBRATION_INTRINSICS_REL
    locmap_path = msight_path / CALIBRATION_LOCMAP_REL
    intrinsics_hash = _sha256(intrinsics_path)
    locmap_hash = _sha256(locmap_path)
    intrinsics_exists = intrinsics_hash is not None
    locmap_exists = locmap_hash is not None
    intrinsics_is_default = intrinsics_hash == _DEFAULT_INTRINSICS_SHA256
    locmap_is_default = locmap_hash == _DEFAULT_LOCMAP_SHA256

    if not intrinsics_exists or not locmap_exists:
        state = "missing"
    elif intrinsics_is_default and locmap_is_default:
        state = "default"
    elif not intrinsics_is_default and not locmap_is_default:
        state = "user_calibrated"
    else:
        state = "partial"  # one file replaced, the other still default -- inconsistent

    return {
        "state": state,
        "intrinsics_exists": intrinsics_exists,
        "locmap_exists": locmap_exists,
        "intrinsics_is_default": intrinsics_is_default,
        "locmap_is_default": locmap_is_default,
    }


# Shared wording for SESSION_STATE hints (prefixed) and the consent summary (plain).
_CALIBRATION_STATE_LABELS = {
    "missing": (
        "calibration=missing (no calibration files found)",
        "missing (no calibration files found — pipeline may fail to start)",
    ),
    "default": (
        "calibration=default (demo calibration, no user upload yet)",
        "default demo calibration (no custom calibration uploaded)",
    ),
    "user_calibrated": (
        "calibration=user-uploaded",
        "your uploaded calibration",
    ),
    "partial": (
        "calibration=partial (inconsistent — one file replaced, one still default)",
        "inconsistent (one file replaced, one still default — re-upload both)",
    ),
}


def calibration_state_label(state: str, *, prefixed: bool) -> str:
    """Word a _calibration_status()['state'] value for display."""
    prefixed_label, plain_label = _CALIBRATION_STATE_LABELS.get(
        state, (f"calibration=unknown ({state})", f"unknown ({state})")
    )
    return prefixed_label if prefixed else plain_label


def _reset_calibration_to_default(msight_path: Path) -> None:
    """Restore the demo calibration via `git show HEAD:<path>` (touches only those two files)."""
    for rel_path in (CALIBRATION_INTRINSICS_REL, CALIBRATION_LOCMAP_REL):
        result = subprocess.run(
            ["git", "show", f"HEAD:{rel_path.as_posix()}"],
            cwd=msight_path, capture_output=True, check=True,
        )
        dest = msight_path / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(result.stdout)


def _check_msight_env(msight_path: Path, overriding_source: bool = False) -> Optional[str]:
    """overriding_source=True skips validating .env's VIDEO_INPUT/RTSP_URL (they're overridden)."""
    env_path = msight_path / ".env"
    if not env_path.is_file():
        return None

    text = env_path.read_text()
    doubled = _DOUBLED_ENV_RE.search(text)
    if doubled:
        key = doubled.group(1)
        return (
            f"MSight_Vision's .env has a malformed line for '{key}' (looks like "
            f"'{key}={key}=...'). Fix that line by hand before starting the pipeline."
        )

    if overriding_source:
        return None

    if re.search(r'^\s*RTSP_URL\s*=\s*\S+', text, re.MULTILINE):
        return None

    video_match = re.search(r'^\s*VIDEO_INPUT\s*=\s*(\S+)', text, re.MULTILINE)
    if video_match and not Path(video_match.group(1)).exists():
        return (
            f"VIDEO_INPUT '{video_match.group(1)}' set in MSight_Vision's .env "
            "does not exist on this host."
        )
    return None


def _friendly_error_from_output(stdout: str, stderr: str) -> Optional[str]:
    combined = f"{stdout}\n{stderr}"
    if _UNDEFINED_VOL_RE.search(combined):
        return (
            "Docker Compose reports an undefined volume — check MSight_Vision's "
            "docker-compose.yml volume definitions."
        )
    if _NVIDIA_ERR_RE.search(combined):
        return (
            "GPU device driver 'nvidia' could not be selected, despite nvidia-smi "
            "reporting a GPU on this host -- the NVIDIA Container Toolkit is likely "
            "missing or misconfigured. (If this host has no GPU at all, this "
            "shouldn't happen -- this repo's own msight_cpu_override.yml should "
            "already be applied automatically; that auto-detection may itself be "
            "the problem.)"
        )
    if _REDIS_PORT_ERR_RE.search(combined):
        return (
            "Redis failed to start — port 6379 is likely already bound by another "
            "process on this host. Stop it and retry."
        )
    return None


_rendered_cpu_override: Optional[Path] = None


def _render_cpu_override() -> Path:
    """Render msight_cpu_override.yml with the absolute Dockerfile.msight-cpu path (cached)."""
    global _rendered_cpu_override
    if _rendered_cpu_override is not None and _rendered_cpu_override.is_file():
        return _rendered_cpu_override
    template = (Path(__file__).parent / "msight_cpu_override.yml").read_text()
    dockerfile = Path(__file__).parent / "Dockerfile.msight-cpu"
    rendered = template.replace("{DOCKERFILE_PATH}", str(dockerfile))
    out = Path(tempfile.gettempdir()) / "msight_cpu_override.rendered.yml"
    out.write_text(rendered)
    _rendered_cpu_override = out
    return out


async def _run_compose(
    msight_path: Path, args: list[str], timeout: int, env: Optional[dict] = None,
    ctx: Optional[Context] = None,
) -> tuple[int, str, str]:
    """Run docker compose, streaming output via ctx.log(); returns the full output."""
    compose_files = ["-f", "docker-compose.yml"]
    # Our own CPU override: MSight_Vision's docker-compose.cpu.yml doesn't clear
    # the GPU reservation (`deploy: {}` is a no-op under Compose merging).
    if not await has_gpu():
        compose_files += ["-f", str(_render_cpu_override())]
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "compose", *compose_files, "--env-file", ".env", *args,
            cwd=str(msight_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
    except FileNotFoundError:
        return -1, "", "Docker is not installed, or not on PATH, on this host."

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    async def _read(stream, buf: list[str]) -> None:
        while True:
            raw = await stream.readline()
            if not raw:
                break
            text = raw.decode(errors="replace").rstrip()
            if not text:
                continue
            buf.append(text)
            if ctx:
                await ctx.log(text)

    try:
        await asyncio.wait_for(
            asyncio.gather(_read(proc.stdout, stdout_lines), _read(proc.stderr, stderr_lines)),
            timeout=timeout,
        )
        await proc.wait()
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "\n".join(stdout_lines), f"Command timed out after {timeout}s: docker compose {' '.join(args)}"

    return proc.returncode, "\n".join(stdout_lines), "\n".join(stderr_lines)


_control_plane_singleton: Optional[MSightControlPlane] = None


def _control_plane() -> MSightControlPlane:
    global _control_plane_singleton
    if _control_plane_singleton is None:
        _control_plane_singleton = MSightControlPlane()
    return _control_plane_singleton


def _default_source_from_env(msight_path: Path) -> tuple[Optional[str], Optional[str]]:
    """(video_input, rtsp_url) from MSight_Vision's .env when neither was given."""
    from dotenv import dotenv_values
    values = dotenv_values(msight_path / ".env")
    return (values.get("VIDEO_INPUT") or None), (values.get("RTSP_URL") or None)


def _resolve_video_source(video_input: Optional[str], rtsp_url: Optional[str], sensor: str) -> tuple[str, dict]:
    """(node_type, config) for video_source, mirroring docker-compose.yml's entrypoint."""
    base = {"publish_topic": f"camera/{sensor}", "sensor_name": sensor}
    if rtsp_url:
        return "video_source_rtsp", {**base, "rtsp_url": rtsp_url}
    # Directory -> mp4_folder; a single file works with msight_launch_rtsp's --url.
    return (
        ("video_source_mp4_folder", {**base, "folder": video_input})
        if Path(video_input).is_dir()
        else ("video_source_rtsp", {**base, "rtsp_url": video_input})
    )


@mcp.tool()
async def start_msight_pipeline(
    video_input: Optional[str] = None,
    rtsp_url: Optional[str] = None,
    sensor_name: Optional[str] = None,
    build: bool = False,
    ctx: Context = None,
) -> str:
    msight_path, err = _get_msight_path()
    if err:
        return error_json(err)

    if video_input and rtsp_url:
        return error_json("Provide exactly one of video_input or rtsp_url, not both.")

    if video_input and not Path(video_input).exists():
        return error_json(f"video_input path '{video_input}' does not exist on this host.")

    if not video_input and not rtsp_url:
        video_input, rtsp_url = _default_source_from_env(msight_path)

    env_err = _check_msight_env(msight_path, overriding_source=bool(video_input or rtsp_url))
    if env_err:
        return error_json(env_err)

    if build:
        # Explicit image build (DockerExecutor itself never builds).
        returncode, stdout, stderr = await _run_compose(msight_path, ["build"], COMPOSE_TIMEOUT_UP, ctx=ctx)
        if returncode != 0:
            friendly = _friendly_error_from_output(stdout, stderr)
            tail = (stdout + stderr)[-2000:]
            return error_json(friendly or f"docker compose build failed:\n{tail}")

    sensor = sensor_name or "gs_mcity_1"
    video_node_type, video_config = _resolve_video_source(video_input, rtsp_url, sensor)

    desired = [
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

    try:
        await _control_plane().recompose(desired)
    except Exception as e:
        msg = str(e)
        friendly = _friendly_error_from_output(msg, "")
        return error_json(friendly or f"Failed to start MSight_Vision pipeline: {msg}")

    return ok_json("MSight_Vision pipeline started.", viewer_url=f"http://{resolve_host()}:{VIEWER_PORT}")


@mcp.tool()
async def stop_msight_pipeline(remove_volumes: bool = False, ctx: Context = None) -> str:
    # remove_volumes is a no-op, kept for schema compatibility.
    msight_path, err = _get_msight_path()
    if err:
        return error_json(err)

    try:
        cp = _control_plane()
        # Explicit names, not recompose([]), so this still works after a server restart.
        for name in FIXED_PIPELINE_NODES:
            await cp.delete_node(name)
    except Exception as e:
        return error_json(f"Failed to stop MSight_Vision pipeline: {e}")

    return ok_json("MSight_Vision pipeline stopped.")


async def _enrich_node_status(cp, nodes: list[dict]) -> list[dict]:
    """Add seconds_since_heartbeat and real liveness (Redis status never expires)."""
    now = time.time()
    out = []
    for n in nodes:
        n = dict(n)
        hb = n.get("last_heartbeat")
        if isinstance(hb, (int, float)):
            n["seconds_since_heartbeat"] = max(0, int(now - hb))
        n["alive"] = await cp.is_alive(n["name"])
        out.append(n)
    return out


@mcp.tool()
async def get_msight_status(ctx: Context = None) -> str:
    msight_path, err = _get_msight_path()
    if err:
        return error_json(err)

    try:
        cp = _control_plane()
        services = await _enrich_node_status(cp, cp.get_status())
    except Exception as e:
        return error_json(f"Could not read MSight_Vision status: {e}")

    extra = {"services": services}
    # viewer_url only when the viewer is actually alive (Redis entries can be stale).
    if any(s.get("name") == "detection_viewer" and s.get("alive") for s in services):
        extra["viewer_url"] = f"http://{resolve_host()}:{VIEWER_PORT}"

    # Precomputed structural problems; the LLM is bad at spotting absences in a list.
    problems = await cp.structural_check()
    if problems:
        extra["problems"] = problems
        extra["next_step"] = (
            "Call diagnose_msight_pipeline for the ranked root cause -- it also measures "
            "live data flow, which catches nodes that are running but stuck."
        )
    elif any(s.get("alive") for s in services):
        # "Up" is not "working".
        extra["data_flow"] = (
            "NOT MEASURED -- this only shows processes are up. If the user asked whether "
            "the pipeline is working or healthy, call diagnose_msight_pipeline before answering."
        )
    return ok_json(**extra)


@mcp.tool()
async def diagnose_msight_pipeline(ctx: Context = None) -> str:
    msight_path, err = _get_msight_path()
    if err:
        return error_json(err)
    try:
        result = await _control_plane().diagnose()
    except Exception as e:
        return error_json(f"Could not diagnose MSight_Vision pipeline: {e}")
    return ok_json(**result)


_LOG_TS_RE = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+')


def _seconds_since_last_log_line(text: str) -> Optional[int]:
    """Parse msight_core's `YYYY-MM-DD HH:MM:SS,ms` log timestamp (a hint only)."""
    for line in reversed(text.splitlines()):
        m = _LOG_TS_RE.match(line)
        if m:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            return max(0, int(time.time() - ts))
    return None


@mcp.tool()
async def get_msight_logs(service: Optional[str] = None, tail: int = 200, ctx: Context = None) -> str:
    msight_path, err = _get_msight_path()
    if err:
        return error_json(err)

    tail = max(1, min(tail, 2000))
    cp = _control_plane()
    try:
        if service:
            combined = await cp.get_logs(service, tail)
            extra = {
                "logs": combined[-8000:],
                "seconds_since_last_line": _seconds_since_last_log_line(combined),
            }
        else:
            names = [n["name"] for n in cp.get_status()]
            per_node = {name: await cp.get_logs(name, tail) for name in names}
            combined = "\n\n".join(f"==> {name} <==\n{text}" for name, text in per_node.items())
            extra = {
                "logs": combined[-8000:],
                "log_freshness": {
                    name: _seconds_since_last_log_line(text) for name, text in per_node.items()
                },
            }
    except Exception as e:
        return error_json(f"Could not fetch logs: {e}")

    return ok_json(**extra)


@mcp.tool()
def check_msight_calibration_status() -> str:
    """Whether real (non-default) calibration files are in place -- see
    _calibration_status for the checksum comparison this wraps."""
    msight_path, err = _get_msight_path()
    if err:
        return error_json(err)

    status = _calibration_status(msight_path)
    messages = {
        "missing": "No calibration files found at all.",
        "default": "Only the shipped demo calibration is present — no user calibration uploaded yet.",
        "user_calibrated": "User-uploaded calibration is active.",
        "partial": "Inconsistent state: one calibration file has been replaced but not the other.",
    }
    return ok_json(messages[status["state"]], **status)
