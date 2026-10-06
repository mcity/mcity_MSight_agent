"""Pluggable execution backends for MSightControlPlane.

An Executor runs an already-built argv and tracks it by name; it knows nothing
about node types or topics. Extra backends can be registered via the
"msight_agent.executors" entry-point group.
"""
import asyncio
import importlib.metadata as metadata
import os
import shlex
import shutil
import signal
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

ENTRY_POINT_GROUP = "msight_agent.executors"
DEFAULT_BACKEND = "supervisor"

LOG_DIR = Path("output/logs/msight_record_archive")
_STARTUP_GRACE_SECONDS = 0.5

_gpu_available: Optional[bool] = None


async def has_gpu() -> bool:
    """Cached nvidia-smi check."""
    global _gpu_available
    if _gpu_available is not None:
        return _gpu_available
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        _gpu_available = False
        return False
    try:
        proc = await asyncio.create_subprocess_exec(
            nvidia_smi, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        _gpu_available = await asyncio.wait_for(proc.wait(), timeout=5) == 0
    except Exception:
        _gpu_available = False
    return _gpu_available


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


_CPU_IMAGE_TAG = "msight-vision-cpu-local"
_CPU_BASE_IMAGE = os.environ.get(
    "MSIGHT_VISION_CPU_BASE_IMAGE", "michigantrafficlab/msight-core:latest-pytorch2.7.1-cpu"
)
_CPU_DOCKERFILE = Path(__file__).resolve().parent / "Dockerfile.msight-cpu"
_cpu_image_ready: Optional[bool] = None


async def _ensure_cpu_image(msight_path: Path) -> str:
    """Build the CPU image from Dockerfile.msight-cpu once per process (no prebuilt CPU tag exists)."""
    global _cpu_image_ready
    if _cpu_image_ready:
        return _CPU_IMAGE_TAG

    inspect = await asyncio.create_subprocess_exec(
        "docker", "image", "inspect", _CPU_IMAGE_TAG,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    if await inspect.wait() == 0:
        _cpu_image_ready = True
        return _CPU_IMAGE_TAG

    build = await asyncio.create_subprocess_exec(
        "docker", "build",
        "-f", str(_CPU_DOCKERFILE),
        "--build-arg", f"BASE_IMAGE={_CPU_BASE_IMAGE}",
        "-t", _CPU_IMAGE_TAG,
        str(msight_path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await build.communicate()
    if build.returncode != 0:
        raise RuntimeError(
            f"Failed to build the CPU-only image ({_CPU_IMAGE_TAG}): "
            f"{out.decode(errors='replace')[-2000:]}"
        )
    _cpu_image_ready = True
    return _CPU_IMAGE_TAG


class DockerExecutor:
    """One `docker run` per node on the prebuilt msight-vision image.

    Never builds, except the CPU fallback image for GPU nodes on GPU-less hosts.
    """

    async def start(self, name: str, cmd: list[str], cwd: Path, env: dict[str, str],
                     needs_gpu: bool = False, mounts: Optional[list[Mount]] = None) -> DockerHandle:
        container_name = f"msight-{name}"
        node_needs_gpu_fallback = needs_gpu and not await has_gpu()
        if node_needs_gpu_fallback:
            image = await _ensure_cpu_image(cwd)
        else:
            image = os.environ.get("MSIGHT_VISION_IMAGE", "michigantrafficlab/msight-vision:latest")

        # Remove any leftover container with this name (--restart precludes --rm).
        cleanup = await asyncio.create_subprocess_exec(
            "docker", "rm", "-f", container_name,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await cleanup.wait()

        args = [
            "docker", "run", "-d",
            "--name", container_name,
            "--network", "host",
            "--restart", "on-failure:3",
        ]
        if needs_gpu and not node_needs_gpu_fallback:
            args += ["--gpus", "all"]
        for host_path, container_path, mode in (mounts or []):
            args += ["-v", f"{host_path}:{container_path}:{mode}"]
        for key, value in env.items():
            args += ["-e", f"{key}={value}"]
        bash_cmd = "exec " + shlex.join(cmd)
        args += [image, "/bin/bash", "-c", bash_cmd]

        proc = await asyncio.create_subprocess_exec(
            *args, cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"docker run failed for '{name}': {stderr.decode(errors='replace').strip()}")
        return DockerHandle(container_name=container_name)

    async def stop(self, handle: DockerHandle) -> None:
        # Graceful stop so the node can deregister, then remove to free the name.
        proc = await asyncio.create_subprocess_exec(
            "docker", "stop", handle.container_name,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        rm = await asyncio.create_subprocess_exec(
            "docker", "rm", "-f", handle.container_name,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await rm.wait()

    async def logs(self, handle: DockerHandle, tail: int = 200) -> str:
        proc = await asyncio.create_subprocess_exec(
            "docker", "logs", "--tail", str(tail), handle.container_name,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        return out.decode(errors="replace")

    async def is_alive(self, handle: DockerHandle) -> bool:
        """Liveness from the Docker daemon; a missing container is not alive."""
        proc = await asyncio.create_subprocess_exec(
            "docker", "inspect", "-f", "{{.State.Running}}", handle.container_name,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        return proc.returncode == 0 and out.decode().strip() == "true"


class SupervisorHandle(ExecutorHandle):
    """`process` when spawned here; only `pid` when recovered from a pidfile."""
    def __init__(self, name: str, process: Optional[asyncio.subprocess.Process] = None,
                 pid: Optional[int] = None):
        self.process = process
        self.pid = pid if pid is not None else (process.pid if process else None)
        self.name = name


def _pid_path(name: str) -> Path:
    return LOG_DIR / f"{name}.pid"


def recover_supervisor_handle(name: str) -> Optional[SupervisorHandle]:
    """Recover a supervisor node's handle from its pidfile after a restart."""
    pid_path = _pid_path(name)
    if not pid_path.is_file():
        return None
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, 0)  # raises if the PID is gone
    except (ValueError, ProcessLookupError, PermissionError):
        return None
    # Guard against PID reuse: the cmdline must still contain this node's name.
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
        if name not in cmdline:
            return None
    except (FileNotFoundError, PermissionError):
        pass  # not on Linux, or /proc unavailable -- fall back to the pid-alive check alone
    return SupervisorHandle(name=name, pid=pid)


class SupervisorExecutor:
    """Runs each node as a detached local subprocess."""

    def __init__(self):
        self._active: dict[str, asyncio.subprocess.Process] = {}

    def _is_alive(self, name: str) -> bool:
        proc = self._active.get(name)
        return proc is not None and proc.returncode is None

    async def start(self, name: str, cmd: list[str], cwd: Path, env: dict[str, str],
                     needs_gpu: bool = False, mounts: Optional[list[Mount]] = None) -> SupervisorHandle:
        # mounts are docker-only; a subprocess already sees the host filesystem.
        if self._is_alive(name):
            return SupervisorHandle(name, process=self._active[name])

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"{name}.log"
        log_file = open(log_path, "wb")
        full_env = os.environ.copy()
        full_env.update(env)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=log_file, stderr=asyncio.subprocess.STDOUT,
                env=full_env, cwd=str(cwd), start_new_session=True,
            )
        finally:
            log_file.close()

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
            except ProcessLookupError:
                pass
            else:
                for _ in range(50):  # ~5s, matching the in-process timeout above
                    try:
                        os.kill(handle.pid, 0)
                    except ProcessLookupError:
                        break
                    await asyncio.sleep(0.1)
                else:
                    try:
                        os.kill(handle.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        self._active.pop(handle.name, None)
        _pid_path(handle.name).unlink(missing_ok=True)

    async def logs(self, handle: SupervisorHandle, tail: int = 200) -> str:
        log_path = LOG_DIR / f"{handle.name}.log"
        if not log_path.is_file():
            return ""
        lines = log_path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-tail:])

    async def is_alive(self, handle: SupervisorHandle) -> bool:
        """Real liveness -- see DockerExecutor.is_alive."""
        if handle.process is not None:
            return handle.process.returncode is None
        if handle.pid is not None:
            try:
                os.kill(handle.pid, 0)
            except ProcessLookupError:
                return False
            else:
                return True  # PermissionError still means the PID exists
        return False


_BUILTIN_EXECUTORS = {
    "docker": DockerExecutor,
    "supervisor": SupervisorExecutor,
}


def resolve_executor(backend_name: Optional[str] = None) -> Executor:
    name = backend_name or os.environ.get("MSIGHT_EXECUTOR_BACKEND", DEFAULT_BACKEND)

    if name in _BUILTIN_EXECUTORS:
        return _BUILTIN_EXECUTORS[name]()

    eps = {ep.name: ep for ep in metadata.entry_points(group=ENTRY_POINT_GROUP)}
    if name in eps:
        return eps[name].load()()

    available = sorted(set(_BUILTIN_EXECUTORS) | set(eps))
    raise ValueError(
        f"MSIGHT_EXECUTOR_BACKEND={name!r} does not match any built-in or "
        f"registered '{ENTRY_POINT_GROUP}' backend. Available: {available}."
    )
