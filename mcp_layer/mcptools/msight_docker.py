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
COMPOSE_TIMEOUT_UP, COMPOSE_TIMEOUT_SHORT = 600, 60  # build ~206s measured locally; ps/logs/down are fast
FIXED_PIPELINE_NODES = ("video_source", "rfdetr_detector", "detection_viewer")

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
    for rel_path in (CALIBRATION_INTRINSICS_REL, CALIBRATION_LOCMAP_REL):
        result = subprocess.run(
            ["git", "show", f"HEAD:{rel_path.as_posix()}"],
            cwd=msight_path, capture_output=True, check=True,
        )
        dest = msight_path / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(result.stdout)


def _check_msight_env(msight_path: Path, overriding_source: bool = False) -> Optional[str]:
    """overriding_source=True means the caller passed video_input/rtsp_url, which
    docker compose applies as process env and takes precedence over --env-file for
    ${VAR} interpolation -- so .env's own static VIDEO_INPUT/RTSP_URL value never
    actually gets used and isn't worth validating."""
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
    """msight_cpu_override.yml's build.dockerfile is a {DOCKERFILE_PATH}
    placeholder -- substituted here with this repo's own Dockerfile.msight-cpu
    (absolute path, since it lives outside MSight_Vision's checkout and the
    override's build.context, so a relative path wouldn't reach it). Rendered
    once per process into a temp file; the source template and the absolute
    path of this file on disk are both fixed for the process lifetime."""
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
    """Runs docker compose, streaming stdout/stderr line-by-line via ctx.log()
    as they arrive -- a `--build` can take minutes, otherwise the user just
    stares at one static message. Still returns the full accumulated text
    for friendly-error matching and the truncated-tail fallback below."""
    compose_files = ["-f", "docker-compose.yml"]
    # Our own override, not MSight_Vision's docker-compose.cpu.yml -- that file's
    # `deploy: {}` doesn't actually clear the base file's GPU device reservation
    # (Compose merges mappings recursively; an empty override is a no-op), and its
    # BASE_IMAGE build arg is a no-op too (Dockerfile-local hardcodes FROM with no
    # ARG). Kept here rather than fixed in MSight_Vision's checkout, which is
    # never modified.
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
    """Neither video_input nor rtsp_url given -- the old docker-compose flow
    fell back to MSight_Vision's own .env values via --env-file interpolation;
    replicate that by reading them directly. Returns (video_input, rtsp_url)."""
    from dotenv import dotenv_values
    values = dotenv_values(msight_path / ".env")
    return (values.get("VIDEO_INPUT") or None), (values.get("RTSP_URL") or None)


def _resolve_video_source(video_input: Optional[str], rtsp_url: Optional[str], sensor: str) -> tuple[str, dict]:
    """(node_type, config) for the video_source node -- mirrors the branching
    docker-compose.yml's own video_source entrypoint script does at runtime."""
    base = {"publish_topic": f"camera/{sensor}", "sensor_name": sensor}
    if rtsp_url:
        return "video_source_rtsp", {**base, "rtsp_url": rtsp_url}
    # A directory plays sequentially via mp4_folder; a single file path is
    # accepted by msight_launch_rtsp itself (its --url also takes a local path).
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
        # Building the image is the one thing DockerExecutor deliberately
        # doesn't do -- still supported here as an explicit, occasional step,
        # reusing the same compose+CPU-override selection as before.
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
    # remove_volumes is now a no-op -- docker run here never creates named
    # volumes to remove -- kept only so the signature/schema stay unchanged.
    msight_path, err = _get_msight_path()
    if err:
        return error_json(err)

    try:
        cp = _control_plane()
        # Explicit names, not recompose([]) -- recompose only diffs against
        # nodes tracked in *this process's* memory, so it silently does
        # nothing (while still returning "ok") if the server restarted since
        # start_msight_pipeline ran. delete_node() falls back to each node's
        # deterministic container name, so this works either way.
        for name in FIXED_PIPELINE_NODES:
            await cp.delete_node(name)
    except Exception as e:
        return error_json(f"Failed to stop MSight_Vision pipeline: {e}")

    return ok_json("MSight_Vision pipeline stopped.")


async def _enrich_node_status(cp, nodes: list[dict]) -> list[dict]:
    """Adds a computed seconds_since_heartbeat, and a real "alive" field
    (is_alive(), not Redis's self-reported status -- that never expires, so
    a node that died hard without deregistering would report RUNNING forever)."""
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
    # Only report a viewer_url when the viewer is actually alive -- a name
    # match against Redis alone isn't enough (a stale ghost entry has the
    # right name but a self-reported, never-expiring status).
    if any(s.get("name") == "detection_viewer" and s.get("alive") for s in services):
        extra["viewer_url"] = f"http://{resolve_host()}:{VIEWER_PORT}"

    # Explicit, precomputed fact rather than something the caller has to
    # notice on its own -- a node missing entirely from `services` (deleted,
    # or crashed without deregistering) produces the exact same symptom as a
    # stalled one, but relying on the reader to spot an absence from a list
    # of what IS present has repeatedly not worked in practice.
    fixed_alive = {name: await cp.is_alive(name) for name in FIXED_PIPELINE_NODES}
    missing = [name for name, alive in fixed_alive.items() if not alive]
    if missing and len(missing) < len(FIXED_PIPELINE_NODES):
        extra["missing_pipeline_nodes"] = missing
        extra["diagnosis"] = (
            f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} NOT running, "
            "even though other fixed-pipeline nodes are still up. This is very likely "
            "the cause of any 'frozen'/'not working'/'no detections' symptom."
        )
    return ok_json(**extra)


_LOG_TS_RE = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+')


def _seconds_since_last_log_line(text: str) -> Optional[int]:
    """msight_core nodes log with a leading `YYYY-MM-DD HH:MM:SS,ms` timestamp
    (confirmed live: `2026-09-08 14:38:22,397 - ... - INFO :: ...`). Used to
    tell a genuinely-stuck node (heartbeat green, but nothing logged in a
    while) apart from one that's just idle between messages -- see
    prompts/msight_reference/diagnosing_stalled_nodes.md for the heuristic."""
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
