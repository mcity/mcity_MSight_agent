import os
import re
import json
import socket
import hashlib
import asyncio
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastmcp import Context

from mcptools import mcp
from host_utils import resolve_host

load_dotenv()

VIEWER_PORT = 9010
COMPOSE_TIMEOUT_UP, COMPOSE_TIMEOUT_SHORT = 600, 60  # build ~206s measured locally; ps/logs/down are fast

# Fixed calibration file locations, matching what rfdetr_config.yaml points at.
CALIBRATION_INTRINSICS_REL = Path("examples/rfdetr/calibration/intrinsics.json")
CALIBRATION_LOCMAP_REL = Path("examples/rfdetr/locmaps/locmap_sip_gs_Fuller_Glazier2_v1.npz")

# SHA256 of the shipped demo calibration files -- distinguishes "still the
# default" from "user uploaded their own" without a separate flag that could
# drift from what's actually on disk.
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
    """Live filesystem + checksum check, called both by the MCP tool below
    and directly (no MCP round-trip) by chat_server.py's per-turn state hint."""
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


# prefixed for SESSION_STATE hints, plain for the consent summary -- single
# source of truth so the two can't drift apart on wording.
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
    """Restore the shipped demo calibration files, overwriting any user
    upload -- called whenever msight_pipeline is freshly (re)selected.
    Uses `git show HEAD:<path>` rather than `git checkout` so this only ever
    touches the two calibration files via a plain write, never the working
    tree, and can't clobber unrelated uncommitted changes in that repo."""
    import subprocess
    for rel_path in (CALIBRATION_INTRINSICS_REL, CALIBRATION_LOCMAP_REL):
        result = subprocess.run(
            ["git", "show", f"HEAD:{rel_path.as_posix()}"],
            cwd=msight_path, capture_output=True, check=True,
        )
        dest = msight_path / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(result.stdout)


def _check_msight_env(msight_path: Path) -> Optional[str]:
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

    if re.search(r'^\s*RTSP_URL\s*=\s*\S+', text, re.MULTILINE):
        return None

    video_match = re.search(r'^\s*VIDEO_INPUT\s*=\s*(\S+)', text, re.MULTILINE)
    if video_match and not Path(video_match.group(1)).exists():
        return (
            f"VIDEO_INPUT '{video_match.group(1)}' set in MSight_Vision's .env "
            "does not exist on this host."
        )
    return None


def _check_redis_port_free() -> Optional[str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        in_use = sock.connect_ex(("127.0.0.1", 6379)) == 0
    finally:
        sock.close()
    if in_use:
        return (
            "Port 6379 is already in use on this host (likely a host redis-server). "
            "Stop it first (e.g. `sudo systemctl stop redis-server`) and retry."
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
            "GPU device driver 'nvidia' could not be selected. The NVIDIA Container "
            "Toolkit is likely missing or misconfigured on this host. Install it, or "
            "run against docker-compose.cpu.yml instead."
        )
    if _REDIS_PORT_ERR_RE.search(combined):
        return (
            "Redis failed to start — port 6379 is likely already bound by another "
            "process on this host. Stop it and retry."
        )
    return None


async def _run_compose(
    msight_path: Path, args: list[str], timeout: int, env: Optional[dict] = None,
    ctx: Optional[Context] = None,
) -> tuple[int, str, str]:
    """Runs docker compose, streaming stdout/stderr line-by-line via ctx.log()
    as they arrive -- a `--build` can take minutes, otherwise the user just
    stares at one static message. Still returns the full accumulated text
    for friendly-error matching and the truncated-tail fallback below."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "compose", "--env-file", ".env", *args,
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


@mcp.tool()
async def start_msight_pipeline(
    video_input: Optional[str] = None,
    rtsp_url: Optional[str] = None,
    sensor_name: Optional[str] = None,
    build: bool = True,
    ctx: Context = None,
) -> str:
    msight_path, err = _get_msight_path()
    if err:
        return json.dumps({"status": "error", "message": err})

    if video_input and rtsp_url:
        return json.dumps({
            "status": "error",
            "message": "Provide exactly one of video_input or rtsp_url, not both.",
        })

    if video_input and not Path(video_input).exists():
        return json.dumps({
            "status": "error",
            "message": f"video_input path '{video_input}' does not exist on this host.",
        })

    env_err = _check_msight_env(msight_path)
    if env_err:
        return json.dumps({"status": "error", "message": env_err})

    redis_err = _check_redis_port_free()
    if redis_err:
        return json.dumps({"status": "error", "message": redis_err})

    compose_env = os.environ.copy()
    if video_input:
        compose_env["VIDEO_INPUT"] = video_input
        compose_env.pop("RTSP_URL", None)
    elif rtsp_url:
        compose_env["RTSP_URL"] = rtsp_url
        compose_env.pop("VIDEO_INPUT", None)
    if sensor_name:
        compose_env["SENSOR_NAME"] = sensor_name

    args = ["up", "-d"] + (["--build"] if build else [])
    returncode, stdout, stderr = await _run_compose(
        msight_path, args, COMPOSE_TIMEOUT_UP, env=compose_env, ctx=ctx
    )

    if returncode != 0:
        friendly = _friendly_error_from_output(stdout, stderr)
        if friendly:
            return json.dumps({"status": "error", "message": friendly})
        tail = (stdout + stderr)[-2000:]
        return json.dumps({"status": "error", "message": f"docker compose up failed:\n{tail}"})

    return json.dumps({
        "status": "ok",
        "message": "MSight_Vision pipeline started.",
        "viewer_url": f"http://{resolve_host()}:{VIEWER_PORT}",
    })


@mcp.tool()
async def stop_msight_pipeline(remove_volumes: bool = False, ctx: Context = None) -> str:
    msight_path, err = _get_msight_path()
    if err:
        return json.dumps({"status": "error", "message": err})

    args = ["down"] + (["-v"] if remove_volumes else [])
    returncode, stdout, stderr = await _run_compose(msight_path, args, COMPOSE_TIMEOUT_SHORT, ctx=ctx)

    if returncode != 0:
        friendly = _friendly_error_from_output(stdout, stderr)
        tail = (stdout + stderr)[-2000:]
        return json.dumps({
            "status": "error",
            "message": friendly or f"docker compose down failed:\n{tail}",
        })

    return json.dumps({"status": "ok", "message": "MSight_Vision pipeline stopped."})


@mcp.tool()
async def get_msight_status(ctx: Context = None) -> str:
    msight_path, err = _get_msight_path()
    if err:
        return json.dumps({"status": "error", "message": err})

    # -a: without it, a crashed service (e.g. video_source on a bad RTSP URL)
    # just disappears from the list instead of showing "exited" -- a broken
    # pipeline would look healthy since the other containers stay up.
    returncode, stdout, stderr = await _run_compose(
        msight_path, ["ps", "-a", "--format", "json"], COMPOSE_TIMEOUT_SHORT, ctx=ctx
    )
    if returncode != 0:
        friendly = _friendly_error_from_output(stdout, stderr)
        tail = (stdout + stderr)[-2000:]
        return json.dumps({
            "status": "error",
            "message": friendly or f"docker compose ps failed:\n{tail}",
        })

    services = []
    stripped = stdout.strip()
    if stripped:
        try:
            parsed = json.loads(stripped)
            services = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            # Older compose versions emit JSON-lines instead of a JSON array.
            for line in stripped.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    services.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return json.dumps({
        "status": "ok",
        "services": services,
        "viewer_url": f"http://{resolve_host()}:{VIEWER_PORT}",
    })


@mcp.tool()
async def get_msight_logs(service: Optional[str] = None, tail: int = 200, ctx: Context = None) -> str:
    msight_path, err = _get_msight_path()
    if err:
        return json.dumps({"status": "error", "message": err})

    tail = max(1, min(tail, 2000))
    args = ["logs", "--no-color", "--tail", str(tail)] + ([service] if service else [])
    returncode, stdout, stderr = await _run_compose(msight_path, args, COMPOSE_TIMEOUT_SHORT, ctx=ctx)

    combined = stdout + stderr
    friendly = _friendly_error_from_output(stdout, stderr)

    if returncode != 0 and not combined.strip():
        return json.dumps({
            "status": "error",
            "message": friendly or "docker compose logs failed with no output.",
        })

    result = {"status": "ok", "logs": combined[-8000:]}
    if friendly:
        result["warning"] = friendly
    return json.dumps(result)


@mcp.tool()
def check_msight_calibration_status() -> str:
    """Whether real (non-default) calibration files are in place -- see
    _calibration_status for the checksum comparison this wraps."""
    msight_path, err = _get_msight_path()
    if err:
        return json.dumps({"status": "error", "message": err})

    status = _calibration_status(msight_path)
    messages = {
        "missing": "No calibration files found at all.",
        "default": "Only the shipped demo calibration is present — no user calibration uploaded yet.",
        "user_calibrated": "User-uploaded calibration is active.",
        "partial": "Inconsistent state: one calibration file has been replaced but not the other.",
    }
    return json.dumps({"status": "ok", "message": messages[status["state"]], **status})
