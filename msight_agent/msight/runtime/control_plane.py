"""MSightControlPlane: the single owner of tracked MSight nodes (no MCP tools here).

Each node runs on the Executor its NodeSpec names (docker or supervisor).
Status is read from the Redis hash MSIGHT:NODES, where every msight_core
node self-registers, so it is backend-agnostic.
"""
import asyncio
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import redis

from msight_agent.msight.runtime.diagnosis import diagnose_from_health, sample_topic_rates, structural_findings
from msight_agent.msight.runtime.executors import (
    DockerHandle, Executor, ExecutorHandle, container_name, gpu_report, inspect_containers,
    recover_supervisor_handle, resolve_executor,
)
from msight_agent.msight.runtime.node_catalog import NODE_CATALOG, resolve_cmd
from msight_agent.msight.runtime.paths import get_msight_path
from msight_agent.msight.runtime.utils import ago, redis_kwargs, run_cmd, topic_list

NODES_REDIS_KEY = "MSIGHT:NODES"  # matches msight_core.nodes.REDIS_NODES_FIELD
_URL_SCHEMES = ("rtsp://", "rtsps://", "http://", "https://", "rtmp://")
_HOST_PROCESS = {"device": "cpu", "note": "host process"}


@dataclass
class _TrackedNode:
    handle: ExecutorHandle
    node_type: str
    invocation: str


def _resolve_msight_path() -> Path:
    path, err = get_msight_path(require_compose=False)
    if err:
        raise RuntimeError(err)
    return path


def _venv_bin(msight_path: Path, name: str) -> Path:
    candidate = msight_path / "venv" / "bin" / name
    if not candidate.is_file():
        raise RuntimeError(
            f"'{name}' not found in {candidate.parent} -- "
            "has MSight_Vision's venv been installed (see its install.sh)?"
        )
    return candidate


def _redis_reachable() -> bool:
    try:
        redis.Redis(**redis_kwargs(), socket_connect_timeout=1).ping()
        return True
    except redis.RedisError:
        return False


def web_port_from_cmd(cmd: str) -> Optional[int]:
    """The port a running web node serves on, from its command line (catalog-driven)."""
    parts = cmd.split()
    for spec in NODE_CATALOG.values():
        if spec.web_port and spec.binary in parts:
            if "--port" in parts and parts.index("--port") + 1 < len(parts):
                return int(parts[parts.index("--port") + 1])
            return spec.web_port
    return None


def _cpu_fallback_reason(gpu: dict) -> str:
    if gpu["usable"]:
        return "the GPU wasn't usable when it started but is now -- restarting it would use the GPU"
    if gpu["hardware"]:
        return (f"{gpu['reason'].rstrip('.')}. NVIDIA hardware IS present ({', '.join(gpu['hardware'])}), "
                "so the driver is the problem, not the machine")
    return gpu["reason"] or "no NVIDIA GPU was found"


class MSightControlPlane:
    def __init__(self):
        self._nodes: dict[str, _TrackedNode] = {}
        self._executors: dict[str, Executor] = {}
        self._redis: Optional[redis.Redis] = None
        # In-memory: lost on mcp_server restart. Every add/remove (chat, dashboard,
        # API) goes through this process, so nothing else needs to write here.
        self._changes: deque[dict] = deque(maxlen=50)

    # --- change log -----------------------------------------------------------

    def _record_change(self, change: str, name: str, node_type: Optional[str] = None,
                       publish_topics: Optional[list[str]] = None) -> None:
        self._changes.append({"at": time.time(), "change": change, "node": name,
                              "node_type": node_type, "publish_topics": publish_topics or []})

    def recent_changes(self, limit: int = 10) -> list[dict]:
        now = time.time()
        return [{"change": c["change"], "node": c["node"], "node_type": c["node_type"],
                 "when": ago(now - c["at"])} for c in list(self._changes)[-limit:]]

    # --- lifecycle --------------------------------------------------------------

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

        tracked = self._nodes.get(name)
        if tracked is not None:
            # Check liveness, not tracking, so an externally-killed node gets restarted.
            if await self._executor_for(tracked.invocation).is_alive(tracked.handle):
                return tracked.handle
            del self._nodes[name]

        msight_path = _resolve_msight_path()
        config = dict(config or {})
        mounts: list[tuple[str, str, str]] = []
        if spec.invocation == "docker":
            raw_value = config.get(spec.mount_config_key) if spec.mount_config_key else None
            if raw_value and not str(raw_value).startswith(_URL_SCHEMES):
                mounts.append((str(raw_value), spec.mount_container_path, "ro"))
                config[spec.mount_config_key] = spec.mount_container_path
            mounts += [(str(msight_path / rel), dest, mode) for rel, dest, mode in spec.fixed_mounts]

        cmd = resolve_cmd(spec, name, config)
        if spec.invocation == "supervisor":
            cmd[0] = str(_venv_bin(msight_path, spec.binary))
            if spec.script_path is not None:
                cmd.insert(1, str(spec.script_path))

        env = {"MSIGHT_EDGE_DEVICE_NAME": os.environ.get("MSIGHT_EDGE_DEVICE_NAME", "mcity_edge")}

        await self._ensure_redis()
        handle = await self._executor_for(spec.invocation).start(
            name, cmd, msight_path, env, needs_gpu=spec.needs_gpu, mounts=mounts)
        self._nodes[name] = _TrackedNode(handle=handle, node_type=node_type, invocation=spec.invocation)
        self._record_change("added", name, node_type)
        return handle

    async def delete_node(self, name: str) -> None:
        registered = self._redis_client().hget(NODES_REDIS_KEY, name)
        existed = registered is not None or await self.is_alive(name)
        try:
            publish_topics = topic_list(json.loads(registered).get("publish_topic")) if registered else []
        except (TypeError, ValueError):
            publish_topics = []

        tracked = self._nodes.pop(name, None)
        if tracked is not None:
            await self._executor_for(tracked.invocation).stop(tracked.handle)
        else:
            # Not tracked (e.g. server restarted): try both recovery paths.
            await self._executor_for("docker").stop(DockerHandle(container_name(name)))
            recovered = recover_supervisor_handle(name)
            if recovered is not None:
                await self._executor_for("supervisor").stop(recovered)

        # Nodes don't reliably deregister on shutdown; clear Redis to avoid ghosts.
        self._redis_client().hdel(NODES_REDIS_KEY, name)
        if existed or tracked is not None:
            self._record_change("removed", name, tracked.node_type if tracked else None, publish_topics)

    async def recompose(self, desired: list[dict]) -> None:
        """Make the tracked set match `desired`: [{"node_type", "name"?, "config"?}]."""
        desired_by_name = {
            d.get("name") or NODE_CATALOG[d["node_type"]].default_name or d["node_type"]: d
            for d in desired
        }
        for name in list(self._nodes):
            if name not in desired_by_name:
                await self.delete_node(name)

        for name, d in desired_by_name.items():
            tracked = self._nodes.get(name)
            if tracked is not None:
                # Keep only a live node of the same type; dead or retyped ones are replaced.
                if (tracked.node_type == d["node_type"]
                        and await self._executor_for(tracked.invocation).is_alive(tracked.handle)):
                    continue
                await self.delete_node(name)
            await self.add_node(d["node_type"], name=name, config=d.get("config"))

    async def _ensure_redis(self) -> None:
        """Start a Redis container only if nothing is listening (Redis isn't in NODE_CATALOG)."""
        if _redis_reachable():
            return
        host, port = redis_kwargs()["host"], redis_kwargs()["port"]
        if host not in ("localhost", "127.0.0.1"):
            raise RuntimeError(f"Redis at {host}:{port} is not reachable, and it isn't a local host to auto-start.")

        code, out = await run_cmd("docker", "run", "-d", "--name", "msight-redis", "--network", "host",
                                  "--restart", "on-failure:3", "redis:7-alpine")
        if code != 0 and "already in use" in out:
            code, out = await run_cmd("docker", "start", "msight-redis")  # left over, stopped
        if code != 0:
            raise RuntimeError(f"Could not start msight-redis: {out.strip()}")

        for _ in range(10):
            if _redis_reachable():
                return
            await asyncio.sleep(0.3)
        raise RuntimeError("Started msight-redis but it never became reachable.")

    # --- status -------------------------------------------------------------------

    def _redis_client(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.Redis(**redis_kwargs(), decode_responses=True)
        return self._redis

    def get_status(self) -> list[dict]:
        """Registry entries from MSIGHT:NODES (self-reported, so they can be stale)."""
        nodes = []
        for name, value in self._redis_client().hgetall(NODES_REDIS_KEY).items():
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                parsed = {"raw": value}
            nodes.append({"name": name, **parsed})
        return sorted(nodes, key=lambda n: n["name"])

    def is_tracked(self, name: str) -> bool:
        """In-memory tracking, with a pidfile fallback for supervisor nodes."""
        return name in self._nodes or recover_supervisor_handle(name) is not None

    async def snapshot(self, names: Iterable[str]) -> dict[str, dict]:
        """{name: {"alive", "device"}} -- one docker call for all containers.
        Survives restarts via container name or pidfile."""
        names = list(names)
        containers = await inspect_containers([container_name(n) for n in names])
        out = {}
        for name in names:
            tracked = self._nodes.get(name)
            info = containers.get(container_name(name))
            if tracked is not None and tracked.invocation != "docker":
                alive = await self._executor_for(tracked.invocation).is_alive(tracked.handle)
                out[name] = {"alive": alive, "device": _HOST_PROCESS}
            elif info is not None:
                out[name] = {"alive": info["running"],
                             "device": {k: info[k] for k in ("device", "image", "gpu_fallback")},
                             "web_port": web_port_from_cmd(info["cmd"])}
            elif tracked is None and (recovered := recover_supervisor_handle(name)) is not None:
                alive = await self._executor_for("supervisor").is_alive(recovered)
                out[name] = {"alive": alive, "device": _HOST_PROCESS}
            else:
                out[name] = {"alive": False, "device": None}
        return out

    async def is_alive(self, name: str) -> bool:
        return (await self.snapshot([name]))[name]["alive"]

    async def structural_check(self, registry: Optional[list[dict]] = None,
                               snap: Optional[dict[str, dict]] = None) -> list[dict]:
        """Cheap (no rate probe) -- registry topics + real liveness only.
        Pass get_status()/snapshot() results to reuse them."""
        registry = self.get_status() if registry is None else registry
        snap = snap or await self.snapshot(info["name"] for info in registry)
        return structural_findings([{
            "name": info["name"],
            "alive": snap[info["name"]]["alive"],
            "publish_topics": topic_list(info.get("publish_topic")),
            "subscribe_topics": topic_list(info.get("subscribe_topic")),
        } for info in registry])

    async def compute_report(self, devices: dict[str, Optional[dict]], include_gpu: bool = False) -> dict:
        """The device each node actually got (from snapshot()), with a warning
        when a GPU node fell back to CPU. The host GPU probe (lspci +
        nvidia-smi) only runs when it explains a fallback or include_gpu."""
        nodes = {name: dev for name, dev in devices.items() if dev is not None}
        fallbacks = [name for name, dev in nodes.items() if dev.get("gpu_fallback")]
        report: dict = {"nodes": nodes, "warnings": []}
        if fallbacks or include_gpu:
            report["gpu"] = gpu = await gpu_report()
            report["warnings"] = [
                f"'{name}' needs a GPU but is running on the CPU-only fallback image "
                f"({_cpu_fallback_reason(gpu)}); expect it to be many times slower."
                for name in fallbacks
            ]
        return report

    # --- diagnosis ------------------------------------------------------------------

    async def node_health(self, max_window: float = 10.0) -> tuple[list[dict], dict[str, dict]]:
        """One verdict per node from liveness + measured topic rates (heartbeats
        alone can't tell starved from dead), plus the snapshot it used.

          DEAD          not running
          UNREGISTERED  running, missing from the Redis registry
          STARVED       running, no input arriving (upstream problem)
          STALLED       input arriving (or a source), publishes nothing
          OK            otherwise
        """
        registry = {n["name"]: n for n in self.get_status()}
        names = sorted(set(registry) | set(self._nodes))
        snap = await self.snapshot(names)

        topics = sorted({t for info in registry.values()
                         for t in topic_list(info.get("publish_topic")) + topic_list(info.get("subscribe_topic"))})
        samples = await asyncio.to_thread(sample_topic_rates, topics, max_window)

        def total(topic_names: list[str], key: str):
            return round(sum(samples.get(t, {}).get(key, 0) for t in topic_names), 2) if topic_names else None

        now = time.time()
        rows = []
        for name in names:
            info = registry.get(name) or {}
            alive = snap[name]["alive"]
            pub, sub = topic_list(info.get("publish_topic")), topic_list(info.get("subscribe_topic"))
            if not alive:
                health = "DEAD"
            elif not info:
                health = "UNREGISTERED"
            elif sub and total(sub, "messages") == 0:
                health = "STARVED"
            elif pub and total(pub, "messages") == 0:
                health = "STALLED"
            else:
                health = "OK"
            hb = info.get("last_heartbeat")
            rows.append({
                "name": name,
                "health": health,
                "alive": alive,
                "self_reported_status": info.get("status"),
                "publish_topics": pub,
                "subscribe_topics": sub,
                "in_rate_hz": total(sub, "rate_hz"),
                "out_rate_hz": total(pub, "rate_hz"),
                "seconds_since_heartbeat": max(0, int(now - hb)) if isinstance(hb, (int, float)) else None,
            })
        return rows, snap

    async def diagnose(self, max_window: float = 10.0) -> dict:
        rows, snap = await self.node_health(max_window)
        result = diagnose_from_health(rows, list(self._changes))
        # Kept out of `summary` on purpose: a CPU fallback is a performance note,
        # and mixing it into the root-cause summary made the agent blame the GPU.
        result["compute"] = await self.compute_report(
            {name: v["device"] for name, v in snap.items()}, include_gpu=True)
        result["recent_changes"] = self.recent_changes()
        return result

    async def get_logs(self, name: str, tail: int = 200) -> str:
        tracked = self._nodes.get(name)
        if tracked is not None:
            return await self._executor_for(tracked.invocation).logs(tracked.handle, tail)
        # Not tracked (e.g. server restarted): pidfile first, since `docker logs`
        # on a missing container returns an error message rather than failing.
        recovered = recover_supervisor_handle(name)
        if recovered is not None:
            return await self._executor_for("supervisor").logs(recovered, tail)
        if container_name(name) in await inspect_containers([container_name(name)]):
            return await self._executor_for("docker").logs(DockerHandle(container_name(name)), tail)
        return f"'{name}' is not tracked by this control plane instance and no matching container or process was found."


_instance: Optional[MSightControlPlane] = None


def get_control_plane() -> MSightControlPlane:
    """The process-wide control plane (node tracking must not be split across instances)."""
    global _instance
    if _instance is None:
        _instance = MSightControlPlane()
    return _instance
