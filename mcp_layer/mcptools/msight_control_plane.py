"""MSightControlPlane -- the one wrapper class msight_docker.py and
msight_record_archive.py delegate to. It is the only thing that knows which
nodes are tracked; it has no @mcp.tool() of its own and is never imported by
mcp_server.py directly (it's plumbing, not a tool module).

add_node / delete_node / recompose dispatch each node to whichever Executor
its NodeSpec names (docker vs supervisor) -- picked per node type, not once
globally, since msight_record_archive.py's whole design point is to avoid
Docker entirely, while the RF-DETR stack needs the image's ML environment.

get_status() bypasses the executor completely: every msight_core node
self-registers into the Redis hash "MSIGHT:NODES" (confirmed literal --
docker-compose.yml's own startup line does `redis-cli hdel MSIGHT:NODES
video_source rfdetr_detector detection_viewer`) regardless of who launched
it, so status is read straight from Redis, backend-agnostic for free.
"""
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from mcptools.msight_node_catalog import NODE_CATALOG, resolve_cmd
from mcptools.msight_executors import (
    DockerHandle, Executor, ExecutorHandle, recover_supervisor_handle, resolve_executor,
)

NODES_REDIS_KEY = "MSIGHT:NODES"  # matches msight_core.nodes.REDIS_NODES_FIELD
_URL_SCHEMES = ("rtsp://", "rtsps://", "http://", "https://", "rtmp://")


def _resolve_msight_path() -> Path:
    load_dotenv(override=True)
    raw = os.environ.get("MSIGHT_VISION_PATH")
    if not raw:
        raise RuntimeError("MSIGHT_VISION_PATH is not set in .env.")
    path = Path(raw)
    if not path.is_dir():
        raise RuntimeError(f"MSIGHT_VISION_PATH ('{path}') does not exist or is not a directory.")
    return path


def _venv_bin(msight_path: Path, name: str) -> Path:
    candidate = msight_path / "venv" / "bin" / name
    if not candidate.is_file():
        raise RuntimeError(
            f"'{name}' not found in {msight_path / 'venv' / 'bin'} -- "
            "has MSight_Vision's venv been installed (see its install.sh)?"
        )
    return candidate


@dataclass
class _TrackedNode:
    handle: ExecutorHandle
    node_type: str
    invocation: str


class MSightControlPlane:
    def __init__(self):
        self._nodes: dict[str, _TrackedNode] = {}
        self._executors: dict[str, Executor] = {}
        self._redis = None

    def _executor_for(self, invocation: str) -> Executor:
        if invocation not in self._executors:
            self._executors[invocation] = resolve_executor(invocation)
        return self._executors[invocation]

    async def add_node(self, node_type: str, name: Optional[str] = None,
                        config: Optional[dict] = None) -> ExecutorHandle:
        if node_type not in NODE_CATALOG:
            raise ValueError(f"Unknown node_type '{node_type}'. Known: {sorted(NODE_CATALOG)}")
        spec = NODE_CATALOG[node_type]
        name = name or spec.default_name or node_type
        config = config or {}

        if name in self._nodes:
            tracked = self._nodes[name]
            # Verify liveness, don't just trust presence -- a node that died
            # externally would otherwise stay "tracked" forever and this
            # idempotency check would skip restarting it.
            if await self._executor_for(tracked.invocation).is_alive(tracked.handle):
                return tracked.handle
            del self._nodes[name]

        msight_path = _resolve_msight_path()
        mounts: list[tuple[str, str, str]] = []
        effective_config = dict(config)
        if spec.mount_config_key and spec.invocation == "docker":
            raw_value = effective_config.get(spec.mount_config_key)
            if raw_value and not str(raw_value).startswith(_URL_SCHEMES):
                mounts.append((str(raw_value), spec.mount_container_path, "ro"))
                effective_config[spec.mount_config_key] = spec.mount_container_path
        if spec.invocation == "docker":
            for rel_path, container_path, mode in spec.fixed_mounts:
                mounts.append((str(msight_path / rel_path), container_path, mode))

        cmd = resolve_cmd(spec, name, effective_config)
        if spec.invocation == "supervisor":
            cmd[0] = str(_venv_bin(msight_path, spec.binary))
            if spec.script_path is not None:
                cmd.insert(1, str(spec.script_path))

        env = {"MSIGHT_EDGE_DEVICE_NAME": os.environ.get("MSIGHT_EDGE_DEVICE_NAME", "mcity_edge")}

        await self._ensure_redis()
        executor = self._executor_for(spec.invocation)
        handle = await executor.start(name, cmd, msight_path, env, needs_gpu=spec.needs_gpu, mounts=mounts)
        self._nodes[name] = _TrackedNode(handle=handle, node_type=node_type, invocation=spec.invocation)
        return handle

    def is_tracked(self, name: str) -> bool:
        """In-memory tracking, plus a cheap synchronous fallback for
        supervisor-backed nodes via their pidfile (recover_supervisor_handle
        -- no subprocess call needed, unlike Docker's equivalent check, so
        this stays sync rather than forcing every caller to await it)."""
        if name in self._nodes:
            return True
        return recover_supervisor_handle(name) is not None

    async def is_alive(self, name: str) -> bool:
        """Real liveness, not just tracking -- is_tracked() can be stale.
        Same dual fallback as delete_node(): tracked in memory, or
        recoverable by deterministic name (Docker) / pidfile (Supervisor)
        after a restart."""
        tracked = self._nodes.get(name)
        if tracked is not None:
            return await self._executor_for(tracked.invocation).is_alive(tracked.handle)
        if await self._executor_for("docker").is_alive(DockerHandle(f"msight-{name}")):
            return True
        recovered = recover_supervisor_handle(name)
        if recovered is not None:
            return await self._executor_for("supervisor").is_alive(recovered)
        return False

    async def delete_node(self, name: str) -> None:
        tracked = self._nodes.pop(name, None)
        if tracked is not None:
            executor = self._executor_for(tracked.invocation)
            await executor.stop(tracked.handle)
        else:
            # Not tracked in memory (e.g. the server restarted) -- try both
            # recovery paths; harmless if neither finds anything.
            try:
                await self._executor_for("docker").stop(DockerHandle(f"msight-{name}"))
            except Exception:
                pass
            recovered = recover_supervisor_handle(name)
            if recovered is not None:
                try:
                    await self._executor_for("supervisor").stop(recovered)
                except Exception:
                    pass
        # Nodes don't reliably deregister from Redis on shutdown -- clear it
        # ourselves so get_status() doesn't report ghosts.
        self._redis_client().hdel(NODES_REDIS_KEY, name)

    async def recompose(self, desired: list[dict]) -> dict:
        """desired: list of {"node_type": str, "name": str | None, "config": dict | None}."""
        desired_by_name: dict[str, dict] = {}
        for d in desired:
            spec = NODE_CATALOG[d["node_type"]]
            name = d.get("name") or spec.default_name or d["node_type"]
            desired_by_name[name] = d

        removed, added, unchanged = [], [], []

        for name in list(self._nodes):
            if name not in desired_by_name:
                await self.delete_node(name)
                removed.append(name)

        for name, d in desired_by_name.items():
            tracked = self._nodes.get(name)
            if tracked is not None and tracked.node_type == d["node_type"]:
                # Verify liveness, don't just trust tracking -- a node that
                # died externally would otherwise stay "unchanged" forever.
                executor = self._executor_for(tracked.invocation)
                if await executor.is_alive(tracked.handle):
                    unchanged.append(name)
                    continue
            if tracked is not None:
                await self.delete_node(name)
                removed.append(name)
            await self.add_node(d["node_type"], name=name, config=d.get("config"))
            added.append(name)

        return {"added": added, "removed": removed, "unchanged": unchanged}

    async def _ensure_redis(self) -> None:
        """redis itself isn't an msight_core node -- it never registers into
        MSIGHT:NODES -- so it's deliberately not in NODE_CATALOG, but every
        node needs it reachable before it can even start. Starts our own
        container only if nothing is already listening; leaves an existing
        Redis (host-installed, or someone else's container) untouched."""
        import redis as redis_lib
        host = os.environ.get("MSIGHT_REDIS_MESSAGE_BROKER_HOST", "localhost")
        port = int(os.environ.get("MSIGHT_REDIS_MESSAGE_BROKER_PORT", 6379))
        try:
            redis_lib.Redis(host=host, port=port, socket_connect_timeout=1).ping()
            return
        except Exception:
            pass

        if host not in ("localhost", "127.0.0.1"):
            raise RuntimeError(f"Redis at {host}:{port} is not reachable, and it isn't a local host to auto-start.")

        proc = await asyncio.create_subprocess_exec(
            "docker", "run", "-d", "--name", "msight-redis",
            "--network", "host", "--restart", "on-failure:3",
            "redis:7-alpine",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0 and b"already in use" not in stderr:
            raise RuntimeError(f"Could not start msight-redis: {stderr.decode(errors='replace').strip()}")

        for _ in range(10):
            try:
                redis_lib.Redis(host=host, port=port, socket_connect_timeout=1).ping()
                return
            except Exception:
                await asyncio.sleep(0.3)
        raise RuntimeError("Started msight-redis but it never became reachable.")

    def _redis_client(self):
        if self._redis is None:
            import redis
            self._redis = redis.Redis(
                host=os.environ.get("MSIGHT_REDIS_MESSAGE_BROKER_HOST", "localhost"),
                port=int(os.environ.get("MSIGHT_REDIS_MESSAGE_BROKER_PORT", 6379)),
                db=int(os.environ.get("MSIGHT_REDIS_MESSAGE_BROKER_DB", 0)),
                decode_responses=True,
            )
        return self._redis

    def get_status(self) -> list[dict]:
        raw = self._redis_client().hgetall(NODES_REDIS_KEY)
        nodes = []
        for name, value in raw.items():
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                parsed = {"raw": value}
            nodes.append({"name": name, **parsed})
        return sorted(nodes, key=lambda n: n["name"])

    async def get_logs(self, name: str, tail: int = 200) -> str:
        tracked = self._nodes.get(name)
        if tracked is not None:
            executor = self._executor_for(tracked.invocation)
            return await executor.logs(tracked.handle, tail)
        # Not tracked in this process's memory -- e.g. the server restarted.
        # Docker containers have a deterministic name (msight-{name}), and
        # supervisor-backed nodes leave a pidfile behind -- either way, logs
        # are still reachable even though we lost the in-memory handle.
        try:
            return await self._executor_for("docker").logs(DockerHandle(f"msight-{name}"), tail)
        except Exception:
            pass
        recovered = recover_supervisor_handle(name)
        if recovered is not None:
            return await self._executor_for("supervisor").logs(recovered, tail)
        return f"'{name}' is not tracked by this control plane instance and no matching container or process was found."
