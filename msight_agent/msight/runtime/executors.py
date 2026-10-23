"""Pluggable execution backends for MSightControlPlane, plus host GPU/container facts.

An Executor runs an already-built argv and tracks it by name; it knows nothing
about node types or topics. Extra backends can be registered via the
"msight_agent.executors" entry-point group.
"""
import asyncio
import os
import shlex
import shutil
import signal
import time
from importlib import metadata
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from msight_agent.msight.runtime.utils import run_cmd
from msight_agent.shared.paths import NODE_LOG_DIR

ENTRY_POINT_GROUP = "msight_agent.executors"
DEFAULT_BACKEND = "supervisor"
_STARTUP_GRACE_SECONDS = 0.5

CPU_IMAGE_TAG = "msight-vision-cpu-local"
_CPU_BASE_IMAGE = os.environ.get(
    "MSIGHT_VISION_CPU_BASE_IMAGE", "michigantrafficlab/msight-core:latest-pytorch2.7.1-cpu"
)
_CPU_DOCKERFILE = Path(__file__).resolve().parent / "Dockerfile.cpu"

_GPU_CACHE_SECONDS = 60
_gpu_report_cache: Optional[tuple[float, dict]] = None
_cpu_image_ready = False


# --- host facts ---------------------------------------------------------------

async def gpu_report() -> dict:
    """{"usable", "hardware", "reason"} -- usable means nvidia-smi works.
    hardware (from lspci) separates "no GPU" from "GPU present, driver broken".
    Cached briefly so a fixed driver is picked up without a restart."""
    global _gpu_report_cache
    if _gpu_report_cache and time.monotonic() - _gpu_report_cache[0] < _GPU_CACHE_SECONDS:
        return _gpu_report_cache[1]

    hardware: list[str] = []
    if shutil.which("lspci"):
        _, out = await run_cmd("lspci", timeout=5)
        hardware = [line.split(": ", 1)[-1] for line in out.splitlines()
                    if "NVIDIA" in line and ("VGA" in line or "3D controller" in line)]

    code, out = await run_cmd("nvidia-smi", "--query-gpu=name", "--format=csv,noheader", timeout=5)
    usable = code == 0
    if code == 127:
        reason = "nvidia-smi is not installed"
    else:
        reason = None if usable else f"nvidia-smi failed: {out.strip().splitlines()[0] if out.strip() else f'exit {code}'}"

    report = {"usable": usable, "hardware": hardware, "reason": reason}
    _gpu_report_cache = (time.monotonic(), report)
    return report


async def has_gpu() -> bool:
    return (await gpu_report())["usable"]


async def inspect_containers(container_names: list[str]) -> dict[str, dict]:
    """Liveness + device for many containers in ONE docker call (status is
    polled by the dashboard). Missing containers are simply absent."""
    if not container_names:
        return {}
    _, out = await run_cmd(
        "docker", "inspect", "-f",
        "{{.Name}}|{{.State.Running}}|{{.Config.Image}}|{{json .HostConfig.DeviceRequests}}|{{join .Config.Cmd \" \"}}",
        *container_names, merge_stderr=False,  # nonzero exit just means some were missing
    )
    result = {}
    for line in out.splitlines():
        name, running, image, requests, cmd = (line.split("|", 4) + ["", "", "", ""])[:5]
        result[name.lstrip("/")] = {
            "running": running == "true",
            "device": "gpu" if '"gpu"' in requests.lower() else "cpu",
            "image": image,
            "gpu_fallback": image.split(":")[0] == CPU_IMAGE_TAG,
            "cmd": cmd,
        }
    return result


async def _ensure_cpu_image(msight_path: Path) -> str:
    """Build the CPU image from Dockerfile.cpu once per process (no prebuilt CPU tag exists)."""
    global _cpu_image_ready
    if not _cpu_image_ready:
        code, _ = await run_cmd("docker", "image", "inspect", CPU_IMAGE_TAG)
        if code != 0:
            code, out = await run_cmd("docker", "build", "-f", str(_CPU_DOCKERFILE),
                                      "--build-arg", f"BASE_IMAGE={_CPU_BASE_IMAGE}",
                                      "-t", CPU_IMAGE_TAG, str(msight_path))
            if code != 0:
                raise RuntimeError(f"Failed to build the CPU-only image ({CPU_IMAGE_TAG}): {out[-2000:]}")
        _cpu_image_ready = True
    return CPU_IMAGE_TAG


# --- executors ----------------------------------------------------------------

class ExecutorHandle:
    """Opaque to callers; each backend defines what it needs to stop/log itself."""


Mount = tuple[str, str, str]  # (host_path, container_path, "ro"|"rw")


@runtime_checkable
class Executor(Protocol):
    async def start(self, name: str, cmd: list[str], cwd: Path, env: dict[str, str],
                     needs_gpu: bool = False, mounts: Optional[list[Mount]] = None) -> ExecutorHandle: ...
    async def stop(self, handle: ExecutorHandle) -> None: ...
    async def logs(self, handle: ExecutorHandle, tail: int = 200) -> str: ...
    async def is_alive(self, handle: ExecutorHandle) -> bool: ...


class DockerHandle(ExecutorHandle):
    def __init__(self, container_name: str):
        self.container_name = container_name


def container_name(node_name: str) -> str:
    return f"msight-{node_name}"


class DockerExecutor:
    """One `docker run` per node on the prebuilt msight-vision image.

    Never builds, except the CPU fallback image for GPU nodes on GPU-less hosts.
    """

    async def start(self, name: str, cmd: list[str], cwd: Path, env: dict[str, str],
                     needs_gpu: bool = False, mounts: Optional[list[Mount]] = None) -> DockerHandle:
        use_gpu = needs_gpu and await has_gpu()
        image = (os.environ.get("MSIGHT_VISION_IMAGE", "michigantrafficlab/msight-vision:latest")
                 if use_gpu or not needs_gpu else await _ensure_cpu_image(cwd))
        cname = container_name(name)

        # Remove any leftover container with this name (--restart precludes --rm).
        await run_cmd("docker", "rm", "-f", cname)

        args = ["docker", "run", "-d", "--name", cname, "--network", "host", "--restart", "on-failure:3"]
        if use_gpu:
            args += ["--gpus", "all"]
        for host_path, container_path, mode in mounts or []:
            args += ["-v", f"{host_path}:{container_path}:{mode}"]
        for key, value in env.items():
            args += ["-e", f"{key}={value}"]
        args += [image, "/bin/bash", "-c", "exec " + shlex.join(cmd)]

        code, out = await run_cmd(*args, cwd=str(cwd))
        if code != 0:
            raise RuntimeError(f"docker run failed for '{name}': {out.strip()}")
        return DockerHandle(cname)

    async def stop(self, handle: DockerHandle) -> None:
        # Graceful stop so the node can deregister, then remove to free the name.
        await run_cmd("docker", "stop", handle.container_name)
        await run_cmd("docker", "rm", "-f", handle.container_name)

    async def logs(self, handle: DockerHandle, tail: int = 200) -> str:
        _, out = await run_cmd("docker", "logs", "--tail", str(tail), handle.container_name)
        return out

    async def is_alive(self, handle: DockerHandle) -> bool:
        return (await inspect_containers([handle.container_name])).get(
            handle.container_name, {}).get("running", False)


class SupervisorHandle(ExecutorHandle):
    """`process` when spawned here; only `pid` when recovered from a pidfile."""
    def __init__(self, name: str, process: Optional[asyncio.subprocess.Process] = None,
                 pid: Optional[int] = None):
        self.name = name
        self.process = process
        self.pid = pid if pid is not None else (process.pid if process else None)


def _pid_path(name: str) -> Path:
    return NODE_LOG_DIR / f"{name}.pid"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # exists, owned by someone else
    return True


def recover_supervisor_handle(name: str) -> Optional[SupervisorHandle]:
    """Recover a supervisor node's handle from its pidfile after a restart."""
    try:
        pid = int(_pid_path(name).read_text().strip())
        os.kill(pid, 0)  # PermissionError: another user's process, so not ours
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return None
    # Guard against PID reuse: the cmdline must still contain this node's name.
    try:
        if name not in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace"):
            return None
    except (FileNotFoundError, PermissionError):
        pass  # /proc unavailable -- fall back to the pid-alive check alone
    return SupervisorHandle(name=name, pid=pid)


class SupervisorExecutor:
    """Runs each node as a detached local subprocess."""

    def __init__(self):
        self._active: dict[str, asyncio.subprocess.Process] = {}

    async def start(self, name: str, cmd: list[str], cwd: Path, env: dict[str, str],
                     needs_gpu: bool = False, mounts: Optional[list[Mount]] = None) -> SupervisorHandle:
        # mounts are docker-only; a subprocess already sees the host filesystem.
        running = self._active.get(name)
        if running is not None and running.returncode is None:
            return SupervisorHandle(name, process=running)

        NODE_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = NODE_LOG_DIR / f"{name}.log"
        with open(log_path, "wb") as log_file:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=log_file, stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, **env}, cwd=str(cwd), start_new_session=True,
            )

        await asyncio.sleep(_STARTUP_GRACE_SECONDS)
        if proc.returncode is not None:
            tail = log_path.read_text(errors="replace")[-1000:] if log_path.is_file() else ""
            raise RuntimeError(f"'{name}' exited immediately (code {proc.returncode}): {tail}")

        self._active[name] = proc
        # Written after the startup check so a pidfile always means the node came up.
        _pid_path(name).write_text(str(proc.pid))
        return SupervisorHandle(name, process=proc)

    async def stop(self, handle: SupervisorHandle) -> None:
        if handle.process is not None:
            handle.process.terminate()
            try:
                await asyncio.wait_for(handle.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                handle.process.kill()
                await handle.process.wait()
        elif handle.pid is not None:
            # Recovered from a pidfile: no Process wrapper, signal directly.
            try:
                os.kill(handle.pid, signal.SIGTERM)
                for _ in range(50):  # ~5s, matching the in-process timeout above
                    if not _pid_alive(handle.pid):
                        break
                    await asyncio.sleep(0.1)
                else:
                    os.kill(handle.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self._active.pop(handle.name, None)
        _pid_path(handle.name).unlink(missing_ok=True)

    async def logs(self, handle: SupervisorHandle, tail: int = 200) -> str:
        log_path = NODE_LOG_DIR / f"{handle.name}.log"
        if not log_path.is_file():
            return ""
        return "\n".join(log_path.read_text(errors="replace").splitlines()[-tail:])

    async def is_alive(self, handle: SupervisorHandle) -> bool:
        if handle.process is not None:
            return handle.process.returncode is None
        return handle.pid is not None and _pid_alive(handle.pid)


_BUILTIN_EXECUTORS = {"docker": DockerExecutor, "supervisor": SupervisorExecutor}


def resolve_executor(backend_name: Optional[str] = None) -> Executor:
    name = backend_name or os.environ.get("MSIGHT_EXECUTOR_BACKEND", DEFAULT_BACKEND)
    if name in _BUILTIN_EXECUTORS:
        return _BUILTIN_EXECUTORS[name]()
    eps = {ep.name: ep for ep in metadata.entry_points(group=ENTRY_POINT_GROUP)}
    if name in eps:
        return eps[name].load()()
    raise ValueError(
        f"MSIGHT_EXECUTOR_BACKEND={name!r} does not match any built-in or registered "
        f"'{ENTRY_POINT_GROUP}' backend. Available: {sorted(set(_BUILTIN_EXECUTORS) | set(eps))}."
    )
