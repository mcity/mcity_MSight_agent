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


def _topic_list(value) -> list[str]:
    """msight_core registers publish_topic as a string, a list, or null."""
    if not value:
        return []
    return [value] if isinstance(value, str) else [t for t in value if t]


_BROKEN = {"DEAD", "STALLED", "UNREGISTERED"}


def _publishers(rows: list[dict]) -> dict[str, list[dict]]:
    by_topic: dict[str, list[dict]] = {}
    for r in rows:
        for t in r.get("publish_topics", []):
            by_topic.setdefault(t, []).append(r)
    return by_topic


def structural_findings(rows: list[dict]) -> list[dict]:
    """Cheap graph checks -- registry topics + liveness only, no rate probe,
    so safe to run on every status call. rows need name, alive,
    publish_topics, subscribe_topics.

    DEAD               a registered/tracked node isn't running
    MISSING_PUBLISHER  a running node subscribes to a topic no registered node
                       publishes (upstream removed, crashed and deregistered,
                       or never started) -- this is how a node deleted from the
                       middle of a pipeline shows up, without knowing its name
    TOPIC_MISMATCH     same, but a published topic name is a near-match
    """
    import difflib
    pubs = _publishers(rows)
    findings = []
    for r in rows:
        if not r["alive"]:
            findings.append({
                "kind": "DEAD", "node": r["name"],
                "evidence": f"'{r['name']}' is registered/tracked but its process/container is not running.",
            })
    for r in rows:
        if not r["alive"]:
            continue
        for t in r.get("subscribe_topics", []):
            if pubs.get(t):
                continue  # has a publisher (alive or dead -- a dead one is its own finding)
            close = difflib.get_close_matches(t, list(pubs), n=1, cutoff=0.75)
            if close:
                findings.append({
                    "kind": "TOPIC_MISMATCH", "node": r["name"], "topic": t,
                    "evidence": (f"'{r['name']}' subscribes to '{t}', which nothing publishes; "
                                 f"the closest published topic is '{close[0]}'."),
                })
            else:
                findings.append({
                    "kind": "MISSING_PUBLISHER", "node": r["name"], "topic": t,
                    "evidence": (f"'{r['name']}' subscribes to '{t}', but no registered node publishes "
                                 "it -- its upstream node was removed, crashed and deregistered, "
                                 "or was never started."),
                })
    return findings


# User-facing wording -- no tool names (base_prompt.txt forbids showing them);
# the model maps these to the right tool calls itself.
_FIXES = {
    "DEAD": "Restart that node with the same settings (or restart the whole pipeline "
            "if it's one of the standard pipeline's nodes).",
    "STALLED": "It's receiving input but producing nothing -- check its logs for an error, "
               "then restart it.",
    "UNREGISTERED": "It's running but never finished registering -- check its logs, and "
                    "restart it if it's stuck starting up.",
    "MISSING_PUBLISHER": "Add back the node that should publish this topic, or restart the "
                         "pipeline if it was one of the standard pipeline's nodes.",
    "TOPIC_MISMATCH": "Remove that node and add it back with the corrected input topic.",
}


def diagnose_from_health(rows: list[dict]) -> dict:
    """Ranked root causes vs downstream effects, from node_health() rows.

    A broken node only counts as a root cause if nothing upstream of it is
    also broken -- msight_core nodes kill themselves after ~15s without input,
    so one failure cascades, and the downstream deaths are effects, not causes.
    STARVED nodes are always effects; each is attributed to the first broken
    node found walking upstream from it (or to its missing publisher)."""
    if not rows:
        return {"verdict": "empty", "summary": "No MSight nodes are registered or tracked.",
                "root_causes": [], "affected": [], "nodes": []}

    by_name = {r["name"]: r for r in rows}
    pubs = _publishers(rows)

    def upstream_broken(r: dict, seen: set) -> Optional[str]:
        for t in r.get("subscribe_topics", []):
            for p in pubs.get(t, []):
                if p["name"] in seen:
                    continue
                seen.add(p["name"])
                deeper = upstream_broken(p, seen)
                if deeper:
                    return deeper
                if p["health"] in _BROKEN:
                    return p["name"]
        return None

    def depth(r: dict, seen: set) -> int:
        ups = [p for t in r.get("subscribe_topics", []) for p in pubs.get(t, []) if p["name"] not in seen]
        return 0 if not ups else 1 + max(depth(p, seen | {p["name"]}) for p in ups)

    root_causes, affected = [], []
    for r in rows:
        if r["health"] in _BROKEN:
            culprit = upstream_broken(r, {r["name"]})
            if culprit:
                affected.append({"node": r["name"], "health": r["health"], "caused_by": culprit})
            else:
                evidence = {
                    "DEAD": f"'{r['name']}' is not running.",
                    "UNREGISTERED": f"'{r['name']}' is running but missing from the node registry.",
                    "STALLED": (f"'{r['name']}' is running and receiving "
                                f"{r['in_rate_hz'] if r['in_rate_hz'] is not None else 'n/a (source)'} msg/s, "
                                "but published nothing during the sample."),
                }[r["health"]]
                root_causes.append({"kind": r["health"], "node": r["name"], "evidence": evidence,
                                    "_depth": depth(r, {r["name"]})})

    for f in structural_findings(rows):
        if f["kind"] in ("MISSING_PUBLISHER", "TOPIC_MISMATCH"):
            root_causes.append({**f, "_depth": depth(by_name[f["node"]], {f["node"]})})

    for r in rows:
        if r["health"] != "STARVED":
            continue
        culprit = upstream_broken(r, {r["name"]})
        if not culprit:
            if any(c["node"] == r["name"] for c in root_causes):
                continue  # its own missing/mismatched publisher is already the root cause
            culprit = "unknown -- see nodes"
        affected.append({"node": r["name"], "health": "STARVED", "caused_by": culprit})

    root_causes.sort(key=lambda c: c.pop("_depth"))
    for c in root_causes:
        c["suggested_fix"] = _FIXES[c["kind"]]

    if not root_causes and not affected:
        summary = f"All {len(rows)} nodes are running and data is flowing."
    elif root_causes:
        top = root_causes[0]
        summary = f"Most likely root cause: {top['evidence']}"
        if affected:
            summary += (f" {len(affected)} downstream node(s) are affected as a consequence: "
                        f"{', '.join(a['node'] for a in affected)}.")
    else:
        summary = "Some nodes are starved of input but no root cause was identified -- see nodes."

    return {
        "verdict": "healthy" if not root_causes and not affected else "degraded",
        "summary": summary,
        "root_causes": root_causes,
        "affected": affected,
        "nodes": rows,
    }


def _sample_topic_rates(topics: list[str], max_window: float) -> dict[str, dict]:
    """Per-topic message count and rate, measured by subscribing (the
    `ros2 topic hz` approach). msight_core's Redis backend publishes on a
    channel named exactly the topic string. Own client, decode_responses=False:
    payloads are binary frames/detections, not text.

    Adaptive window: stops as soon as every topic has delivered a message, else
    at max_window. A fixed short window misreads slow-but-healthy topics as
    silent -- RF-DETR on this host emits one detection every ~4.5s, so a 2s
    window flapped between 0 and 1 messages (confirmed live). max_window must
    exceed the slowest healthy interval."""
    import time
    import redis
    if not topics:
        return {}
    client = redis.Redis(
        host=os.environ.get("MSIGHT_REDIS_MESSAGE_BROKER_HOST", "localhost"),
        port=int(os.environ.get("MSIGHT_REDIS_MESSAGE_BROKER_PORT", 6379)),
        db=int(os.environ.get("MSIGHT_REDIS_MESSAGE_BROKER_DB", 0)),
    )
    pubsub = client.pubsub(ignore_subscribe_messages=True)
    counts = dict.fromkeys(topics, 0)
    start = time.monotonic()
    try:
        pubsub.subscribe(*topics)
        deadline = start + max_window
        while (remaining := deadline - time.monotonic()) > 0:
            msg = pubsub.get_message(timeout=remaining)
            if msg and msg["type"] == "message":
                channel = msg["channel"].decode(errors="replace")
                if channel in counts:
                    counts[channel] += 1
                    if all(counts.values()):
                        # Keep listening to at least 1s so an early lucky
                        # message doesn't yield a wildly inflated rate (saw
                        # 11 Hz from 0.1s for a 0.2 Hz topic).
                        deadline = min(deadline, max(start + 1.0, time.monotonic()))
    finally:
        pubsub.close()
        client.close()
    elapsed = max(time.monotonic() - start, 1e-6)
    return {
        t: {"messages": c, "rate_hz": round(c / elapsed, 2), "window_s": round(elapsed, 1)}
        for t, c in counts.items()
    }


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

    async def node_health(self, max_window: float = 10.0) -> list[dict]:
        """One health verdict per node, from real liveness plus measured topic
        rates rather than heartbeats alone. msight_core only heartbeats every N
        *messages*, so a starved downstream node goes stale exactly like a dead
        one -- and with its default action_on_error="stop" it then kills itself,
        cascading one failure down the pipeline. Rates tell the two apart:

          DEAD          process/container not running
          UNREGISTERED  running, but missing from the Redis registry
          STARVED       running, but nothing arrives on its input topic
                        (an upstream problem -- a symptom, not the cause)
          STALLED       input arriving (or it's a source), but it publishes nothing
                        (this node itself is broken)
          OK            otherwise
        """
        import time
        registry = {n["name"]: n for n in self.get_status()}
        names = sorted(set(registry) | set(self._nodes))

        topics: set[str] = set()
        for info in registry.values():
            topics.update(_topic_list(info.get("publish_topic")))
            topics.update(_topic_list(info.get("subscribe_topic")))
        samples = await asyncio.to_thread(_sample_topic_rates, sorted(topics), max_window)

        def _sum(topic_names: list[str], key: str):
            if not topic_names:
                return None
            return round(sum(samples.get(t, {}).get(key, 0) for t in topic_names), 2)

        now = time.time()
        out = []
        for name in names:
            info = registry.get(name)
            alive = await self.is_alive(name)
            pub = _topic_list(info.get("publish_topic")) if info else []
            sub = _topic_list(info.get("subscribe_topic")) if info else []
            in_msgs, out_msgs = _sum(sub, "messages"), _sum(pub, "messages")

            if not alive:
                health = "DEAD"
            elif info is None:
                health = "UNREGISTERED"
            elif sub and in_msgs == 0:
                health = "STARVED"
            elif pub and out_msgs == 0:
                health = "STALLED"
            else:
                health = "OK"

            hb = info.get("last_heartbeat") if info else None
            out.append({
                "name": name,
                "health": health,
                "alive": alive,
                "self_reported_status": info.get("status") if info else None,
                "publish_topics": pub,
                "subscribe_topics": sub,
                "in_rate_hz": _sum(sub, "rate_hz"),
                "out_rate_hz": _sum(pub, "rate_hz"),
                "seconds_since_heartbeat": max(0, int(now - hb)) if isinstance(hb, (int, float)) else None,
            })
        return out

    async def diagnose(self, max_window: float = 10.0) -> dict:
        return diagnose_from_health(await self.node_health(max_window))

    async def structural_check(self) -> list[dict]:
        """Cheap (no rate probe) -- registry topics + real liveness only."""
        rows = []
        for info in self.get_status():
            rows.append({
                "name": info["name"],
                "alive": await self.is_alive(info["name"]),
                "publish_topics": _topic_list(info.get("publish_topic")),
                "subscribe_topics": _topic_list(info.get("subscribe_topic")),
            })
        return structural_findings(rows)

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
