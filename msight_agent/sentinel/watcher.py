"""Fault detection for a running MSight deployment -- plain code, no LLM.

Each signal becomes an Incident on a queue:

  docker events   oom/kill/die of msight-* containers, announced ~2s after the death
                  and grouped until quiet, so Docker's own restart attempts land in
                  one incident
  pidfiles        host-process (supervisor) nodes whose process has died
  heartbeats      a node whose heartbeat stops while its inputs are fine (frozen or
                  hung), against a threshold learned per node
  /diagnose       faults the others can't see: a node that heartbeats but publishes
                  nothing (two bad sweeps in a row); also notices recovery
  ports           the agent's own services (mcp_server, msight_api)

Every stage is published to the live feed (msight/runtime/incidents.py) for the chat.
"""
import asyncio
import json
import os
import socket
import time
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import redis

from msight_agent.msight.runtime.incidents import recent_incidents
from msight_agent.sentinel.incident_log import append_event
from msight_agent.shared.paths import NODE_LOG_DIR

CONTAINER_PREFIX = "msight-"
NODES_REDIS_KEY = "MSIGHT:NODES"  # the msight_core node registry (see msight/runtime/control_plane.py)
SETTLE_SECONDS = 12      # Docker's on-failure restarts of a node come ~7s apart
ANNOUNCE_GRACE_SECONDS = 2   # a deliberate stop removes the container well within this
HEARTBEAT_STALE_SECONDS = 5  # msight_core nodes refresh their registry heartbeat ~1/s
OPEN_SWEEP_SECONDS = 10  # health sweep while an incident is open, to notice recovery quickly
COOLDOWN_SECONDS = 600   # a node just reported isn't reported again by the sweep
SERVICES = {"mcp_server": 8000, "msight_api": 8003}
LOG_TAIL_LINES = 40
_OPENING_ACTIONS = {"oom", "kill", "die"}  # start/create/update alone are normal lifecycle
_SIGNALS = {"9": "SIGKILL", "15": "SIGTERM", "2": "SIGINT", "1": "SIGHUP"}

Say = Callable[[str], None]


@dataclass
class Incident:
    kind: str      # CONTAINER_OOM | CONTAINER_KILLED | CONTAINER_CRASHED | CONTAINER_EXITED
    node: str      # | PROCESS_DIED | PIPELINE_DEGRADED | SERVICE_DOWN
    summary: str
    first_event_at: float
    detected_at: float = field(default_factory=time.time)
    timeline: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"INC-{time.strftime('%Y%m%d-%H%M%S', time.localtime(self.detected_at))}-{self.node}"

    def to_dict(self) -> dict:
        return {"id": self.id, **asdict(self)}


def clock(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


# --- container events (pure, unit-tested) ----------------------------------------------

def parse_docker_event(line: str) -> Optional[tuple[str, dict]]:
    """(container name, event) for an msight-* container event line, else None."""
    try:
        raw = json.loads(line)
    except ValueError:
        return None
    attrs = (raw.get("Actor") or {}).get("Attributes") or {}
    name = attrs.get("name", "")
    if raw.get("Type") != "container" or not name.startswith(CONTAINER_PREFIX):
        return None
    exit_code = attrs.get("exitCode")
    signal = attrs.get("signal")
    return name, {
        "key": f"{raw.get('id')}:{raw.get('Action')}:{raw.get('timeNano')}",
        "time_ns": raw.get("timeNano", 0),
        "at": raw.get("timeNano", 0) / 1e9 or time.time(),
        "action": raw.get("Action", "").split(":")[0],
        "exit_code": int(exit_code) if exit_code not in (None, "") else None,
        "signal": _SIGNALS.get(signal, signal) if signal else None,
    }


class EventReplayGuard:
    """Reconnecting with `docker events --since <last event>` replays the overlap; this
    drops those repeats so no event is lost in a reconnect gap and none is seen twice."""

    def __init__(self, memory: int = 1000):
        self.last_ns = 0
        self._seen: deque[str] = deque(maxlen=memory)

    def since_args(self) -> list[str]:
        if not self.last_ns:
            return []  # first connection: live events only, no history
        return ["--since", f"{self.last_ns // 1_000_000_000}.{self.last_ns % 1_000_000_000:09d}"]

    def is_new(self, event: dict) -> bool:
        if event["key"] in self._seen:
            return False
        self._seen.append(event["key"])
        self.last_ns = max(self.last_ns, event["time_ns"])
        return True


def describe_event(event: dict) -> str:
    extra = "".join(f" {k}={event[k]}" for k in ("exit_code", "signal") if event.get(k) is not None)
    return f"{clock(event['at'])} {event['action']}{extra}"


class EventGrouper:
    """Collects one container's events until none arrive for `settle` seconds."""

    def __init__(self, settle: float = SETTLE_SECONDS):
        self.settle = settle
        self._open: dict[str, list[dict]] = {}

    def add(self, name: str, event: dict) -> None:
        if event["action"] == "destroy":
            # Removed on purpose (pipeline stop, node removal, restore): not a fault.
            self._open.pop(name, None)
        elif name in self._open:
            self._open[name].append(event)
        elif event["action"] in _OPENING_ACTIONS:
            self._open[name] = [event]

    def is_open(self, name: str) -> bool:
        return name in self._open

    def announceable(self, now: float, grace: float = ANNOUNCE_GRACE_SECONDS) -> list[tuple[str, list[dict]]]:
        """Open groups whose first death is at least `grace` old -- long enough for a
        deliberate stop's `docker rm` (destroy) to have cancelled it."""
        out = []
        for name, events in self._open.items():
            deaths = [e["at"] for e in events if e["action"] in ("die", "oom")]
            if deaths and now - deaths[0] >= grace:
                out.append((name, list(events)))
        return out

    def settled(self, now: float) -> list[tuple[str, list[dict]]]:
        done = [(n, evs) for n, evs in self._open.items() if now - evs[-1]["at"] >= self.settle]
        for name, _ in done:
            del self._open[name]
        return done


def describe_failure(events: list[dict], state: dict) -> tuple[str, str]:
    """(kind, what happened) from a container's events so far and `docker inspect` (may be {})."""
    deaths = [e for e in events if e["action"] == "die"]
    ooms = sum(e["action"] == "oom" for e in events)
    # A "kill" event only comes from the Docker API (docker kill/stop); the kernel's
    # OOM killer produces "oom" + "die" with no "kill".
    api_signals = [e["signal"] for e in events if e["action"] == "kill" and e.get("signal")]
    exit_code = deaths[-1]["exit_code"] if deaths else state.get("ExitCode")

    if ooms or state.get("OOMKilled"):
        kind, what = "CONTAINER_OOM", f"was killed by the kernel OOM killer ({max(ooms, 1)}x)"
    elif api_signals:
        kind, what = "CONTAINER_KILLED", f"was sent {api_signals[-1]} through the Docker API (docker kill/stop)"
    elif exit_code in (137, 143):
        kind, what = "CONTAINER_KILLED", f"was killed by a signal (exit {exit_code}), not by the OOM killer"
    elif exit_code:
        kind, what = "CONTAINER_CRASHED", f"crashed (exit code {exit_code})"
    else:
        kind, what = "CONTAINER_EXITED", "exited with code 0 although nothing removed it"
    return kind, what


def classify_container(name: str, events: list[dict], state: dict) -> tuple[str, str]:
    """(kind, one-line summary) from a container's grouped events and final `docker inspect`."""
    node = name.removeprefix(CONTAINER_PREFIX)
    kind, what = describe_failure(events, state)
    restarts = sum(e["action"] == "start" for e in events)
    if state.get("Running"):
        outcome = f"Docker restarted it {restarts}x and it is running again" if restarts else "it is running"
    elif restarts:
        outcome = f"Docker restarted it {restarts}x, then gave up -- it is DOWN"
    else:
        outcome = "Docker did not restart it -- it is DOWN"
    return kind, f"Node '{node}' {what}; {outcome}."


# --- pipeline sweep (pure, unit-tested) -------------------------------------------------

def degraded_key(diagnosis: dict) -> Optional[frozenset]:
    """Identity of a degraded verdict's root causes, or None when not degraded."""
    if diagnosis.get("verdict") != "degraded":
        return None
    # A missing publisher is reported under the starved subscriber's name; the node to
    # bring back (and the one Fix-it should restart) is the removed publisher.
    return frozenset((c.get("kind"), c.get("removed_publisher") or c.get("node") or c.get("topic") or "?")
                     for c in diagnosis.get("root_causes") or [])


def compact_diagnosis(diagnosis: dict) -> dict:
    keep = ("name", "health", "alive", "in_rate_hz", "out_rate_hz", "seconds_since_heartbeat")
    return {
        "verdict": diagnosis.get("verdict"),
        "summary": diagnosis.get("summary"),
        "root_causes": diagnosis.get("root_causes"),
        "affected": diagnosis.get("affected"),
        "nodes": [{k: n.get(k) for k in keep} for n in diagnosis.get("nodes") or []],
    }


class HeartbeatWatch:
    """Nodes that stopped heart-beating while their inputs are fine -- frozen or hung.

    msight_core refreshes a subscriber's registry heartbeat inside its message loop,
    every 10 messages, so (1) a starved node goes quiet too: a stale node whose
    upstream publisher is stale, gone, or only just back is a symptom, not reported; and (2) the
    normal gap depends on the node's rate (seconds on GPU, ~45s for RF-DETR on CPU),
    so each node's threshold is learned. Reading the registry is far cheaper than
    subscribing to the topics (the camera topic alone is ~6 MB/s). The first update
    only learns which entries are already stale (ghosts of earlier runs)."""

    def __init__(self, stale_after: float = HEARTBEAT_STALE_SECONDS, factor: float = 3.0,
                 unlearned: float = 30.0, startup: float = 60.0):
        self.stale_after, self.factor, self.unlearned, self.startup = stale_after, factor, unlearned, startup
        self._last: dict[str, float] = {}
        self._gaps: dict[str, deque] = {}
        self._stale: set[str] = set()
        self._first_hb: dict[str, float] = {}   # heartbeat when a node (re)appeared healthy
        self._ok_since: dict[str, float] = {}   # when that heartbeat first advanced: proven working
        self._reported: set[str] = set()
        self._seeded = False

    def threshold(self, node: str) -> float:
        gaps = sorted(self._gaps.get(node) or [])
        if len(gaps) < 3:
            return max(self.stale_after, self.unlearned)
        return max(self.stale_after, self.factor * gaps[len(gaps) // 2])

    def update(self, nodes: dict[str, dict], now: float) -> list[str]:
        """Nodes newly found stuck. `nodes`: {name: {"heartbeat", "publish", "subscribe"}}."""
        for name, info in nodes.items():
            hb = info.get("heartbeat")
            if not isinstance(hb, (int, float)):
                continue
            previous = self._last.get(name)
            if previous is not None and 0 < hb - previous < 600:
                self._gaps.setdefault(name, deque(maxlen=20)).append(hb - previous)
            self._last[name] = hb
        def limit(name: str) -> float:
            # Until its heartbeat first advances, a (re)started node may still be loading.
            proven = name in self._ok_since
            return self.threshold(name) if proven else max(self.threshold(name), self.startup)

        was_stale = self._stale
        self._stale = {name for name, info in nodes.items() if isinstance(info.get("heartbeat"), (int, float))
                       and now - info["heartbeat"] > limit(name)}
        # A node that (re)registers or comes back from stale gets a fresh heartbeat before
        # it does any work (a restarted detector still has to load its model), so it only
        # counts as working once its heartbeat advances -- every 10 messages it processes.
        # A running node that merely goes stale keeps its status (and normal threshold).
        for name in set(self._first_hb) - set(nodes):
            self._first_hb.pop(name, None)
            self._ok_since.pop(name, None)
        for name, info in nodes.items():
            hb = info.get("heartbeat")
            if not isinstance(hb, (int, float)) or name in self._stale:
                continue
            if name in was_stale:
                self._first_hb[name] = hb
                self._ok_since.pop(name, None)
            elif hb > self._first_hb.setdefault(name, hb):
                self._ok_since.setdefault(name, now)

        publishers: dict[str, list[str]] = {}
        for name, info in nodes.items():
            for topic in info.get("publish") or []:
                publishers.setdefault(topic, []).append(name)

        def upstream_down(publisher: str, node: str) -> bool:
            # Stale, not yet proven working, or working again so recently that its
            # data may not have reached `node` yet.
            since = self._ok_since.get(publisher)
            return publisher in self._stale or since is None or now - since < self.threshold(node)

        def starved(name: str) -> bool:
            return any(not publishers.get(t) or any(upstream_down(p, name) for p in publishers[t])
                       for t in nodes[name].get("subscribe") or [])

        if not self._seeded:
            self._seeded, self._reported = True, set(self._stale)
            return []
        self._reported &= self._stale  # forget nodes that recovered
        newly = sorted(n for n in self._stale if n not in self._reported and not starved(n))
        self._reported |= set(newly)
        return newly

    def is_stale(self, node: str) -> bool:
        return node in self._stale


def _topics(value) -> list[str]:
    """msight_core registers topics as a string, a list, or null."""
    if not value:
        return []
    return [value] if isinstance(value, str) else [t for t in value if t]


def nodes_from_registry(rows: dict) -> dict[str, dict]:
    """{node: {"heartbeat", "publish", "subscribe"}} from an HGETALL of the node registry."""
    out = {}
    for name, value in rows.items():
        name = name.decode() if isinstance(name, bytes) else name
        try:
            info = json.loads(value)
        except (TypeError, ValueError):
            info = {}
        info = info if isinstance(info, dict) else {}
        out[name] = {"heartbeat": info.get("last_heartbeat"), "publish": _topics(info.get("publish_topic")),
                     "subscribe": _topics(info.get("subscribe_topic"))}
    return out


# --- the watcher ----------------------------------------------------------------------

async def _run(*argv: str, timeout: float = 15) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return -1, f"timed out after {timeout:.0f}s"
    return proc.returncode, out.decode(errors="replace")


async def inspect_container(name: str) -> Optional[dict]:
    code, out = await _run("docker", "inspect", name)
    if code != 0:
        return None
    raw = json.loads(out)[0]
    state, host = raw.get("State") or {}, raw.get("HostConfig") or {}
    return {
        **{k: state.get(k) for k in ("Status", "Running", "Paused", "OOMKilled", "ExitCode",
                                     "Error", "StartedAt", "FinishedAt")},
        "RestartCount": raw.get("RestartCount"),
        "RestartPolicy": host.get("RestartPolicy"),
        "MemoryLimitBytes": host.get("Memory") or None,
        "Image": (raw.get("Config") or {}).get("Image"),
    }


def _tail_file(path, lines: int = LOG_TAIL_LINES) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError as e:
        return f"(could not read {path}: {e})"


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def _container_condition(state: Optional[dict]) -> str:
    if state is None:
        return "it has no container (host process or removed)"
    if state.get("Paused"):
        return "Docker reports its container as PAUSED"
    return "its container is still running" if state.get("Running") else "its container is not running"


Publish = Callable[..., None]


class Watcher:
    """Turns the signals above into incidents. Each incident is announced to the live
    feed as soon as it's seen (`publish`), then queued for investigation once its
    facts are settled, and closed when the pipeline is healthy again."""

    def __init__(self, queue: "asyncio.Queue[Incident]", say: Say, publish: Optional[Publish] = None,
                 redis_settings: Optional[dict] = None, api_url: str = "http://localhost:8003",
                 sweep_seconds: float = 30):
        self.queue, self.say, self.api_url, self.sweep_seconds = queue, say, api_url, sweep_seconds
        self.publish: Publish = publish or (lambda *args, **kwargs: None)
        self._redis = redis.Redis(**redis_settings, socket_timeout=2) if redis_settings else None
        self.grouper = EventGrouper()
        self.replay = EventReplayGuard()
        self.heartbeats = HeartbeatWatch()
        self.open: dict[str, Incident] = {}            # announced, not yet resolved/dismissed
        self._announced: dict[str, Incident] = {}      # container name -> its provisional incident
        self._classified: set[str] = set()             # incident ids whose facts are settled
        self._reported_at: dict[str, float] = {}       # node -> when last reported
        self._dead_pids: set[tuple[str, int]] = set()
        self._down_counts = {s: 0 for s in SERVICES}
        self._down_reported: set[str] = set()
        self._pending_key: Optional[frozenset] = None
        self._pending_count = 0
        self._reported_key: Optional[frozenset] = None

    def close_orphans(self) -> None:
        """Incidents left open by a previous sentinel run would stay open in the chat
        forever (this run doesn't know them); close them and re-detect from scratch."""
        if self._redis is None:
            return
        try:
            orphans = recent_incidents(self._redis)
        except redis.RedisError:
            return
        for state in orphans:
            self.publish(state["id"], "dismissed",
                         text="The monitor restarted; if the problem is still there it will be reported again.")
        if orphans:
            self.say(f"[watch] closed {len(orphans)} incident(s) left open by a previous run")

    async def run(self) -> None:
        self.close_orphans()
        await asyncio.gather(self._docker_events(), self._settle_loop(), self._pidfile_loop(),
                             self._heartbeat_loop(), self._sweep_loop())

    # incident lifecycle -----------------------------------------------------------------

    def _announce(self, incident: Incident) -> None:
        """Tell the chat right away, before the facts are settled."""
        self.open[incident.id] = incident
        self._reported_at[incident.node] = time.time()
        self.publish(incident.id, "detected", node=incident.node, kind=incident.kind,
                     summary=incident.summary, first_event_at=incident.first_event_at)
        self.say(f"[detect] {incident.node}: {incident.summary}")

    def _emit(self, incident: Incident) -> None:
        """Facts settled: publish them and queue the investigation."""
        if incident.id not in self.open:
            self._announce(incident)
        self._classified.add(incident.id)
        self._reported_at[incident.node] = time.time()
        self.publish(incident.id, "classified", kind=incident.kind, summary=incident.summary,
                     timeline=incident.timeline)
        self.queue.put_nowait(incident)

    def _close(self, incident: Incident, stage: str, text: str) -> None:
        self.open.pop(incident.id, None)
        self.publish(incident.id, stage, text=text)
        append_event(incident.id, stage, text=text)
        self.say(f"[{stage}] {incident.id}: {text}")

    def _recently_reported(self, node: str) -> bool:
        return time.time() - self._reported_at.get(node, 0) < COOLDOWN_SECONDS

    # docker events --------------------------------------------------------------------

    async def _docker_events(self) -> None:
        while True:
            proc = await asyncio.create_subprocess_exec(
                "docker", "events", "--filter", "type=container", "--format", "{{json .}}",
                *self.replay.since_args(), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
            async for raw in proc.stdout:
                parsed = parse_docker_event(raw.decode(errors="replace"))
                if parsed is None or not self.replay.is_new(parsed[1]):
                    continue
                name, event = parsed
                self.grouper.add(name, event)
                if event["action"] == "destroy" and name in self._announced:
                    self._close(self._announced.pop(name), "dismissed",
                                "The container was removed on purpose (pipeline stop or node removal).")
            await proc.wait()
            self.say("[watch] docker events stream ended; reconnecting in 5s")
            await asyncio.sleep(5)

    async def _settle_loop(self) -> None:
        while True:
            await asyncio.sleep(0.5)
            await self._settle_once(time.time())

    async def _settle_once(self, now: float) -> None:
        """Announce deaths past the grace period; classify groups that went quiet."""
        for name, events in self.grouper.announceable(now):
            if name not in self._announced:
                node = name.removeprefix(CONTAINER_PREFIX)
                kind, what = describe_failure(events, {})
                incident = Incident(kind=kind, node=node, first_event_at=events[0]["at"],
                                    summary=f"Node '{node}' {what}. Checking whether Docker brings it back...",
                                    timeline=[describe_event(e) for e in events])
                self._announced[name] = incident
                self._announce(incident)
        for name, events in self.grouper.settled(now):
            incident = self._announced.pop(name, None)
            state = await inspect_container(name)
            if state is None:
                if incident:
                    self._close(incident, "dismissed", "The container was removed on purpose.")
                continue
            kind, summary = classify_container(name, events, state)
            _, logs = await _run("docker", "logs", "--tail", str(LOG_TAIL_LINES), name)
            incident = incident or Incident(kind=kind, node=name.removeprefix(CONTAINER_PREFIX),
                                            summary=summary, first_event_at=events[0]["at"])
            incident.kind, incident.summary = kind, summary
            incident.timeline = [describe_event(e) for e in events]
            incident.evidence = {"container_state": state, "container_log_tail": logs}
            self._emit(incident)

    # host-process nodes -----------------------------------------------------------------

    async def _pidfile_loop(self) -> None:
        while True:
            for pidfile in NODE_LOG_DIR.glob("*.pid") if NODE_LOG_DIR.is_dir() else []:
                try:
                    pid = int(pidfile.read_text().strip())
                except (OSError, ValueError):
                    continue
                # The executor deletes the pidfile on a deliberate stop, so a pidfile
                # whose process is gone means the node died on its own.
                if (pidfile.stem, pid) in self._dead_pids or _pid_alive(pid):
                    continue
                self._dead_pids.add((pidfile.stem, pid))
                now = time.time()
                self._emit(Incident(
                    kind="PROCESS_DIED", node=pidfile.stem, first_event_at=now,
                    summary=f"Host-process node '{pidfile.stem}' (pid {pid}) is gone; nothing restarts these.",
                    timeline=[f"{clock(now)} pid {pid} found dead (pidfile still present)"],
                    evidence={"pidfile": str(pidfile), "log_tail": _tail_file(NODE_LOG_DIR / f"{pidfile.stem}.log")}))
            await asyncio.sleep(2)

    # frozen / hung nodes ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        if self._redis is None:
            return
        while True:
            await asyncio.sleep(1)
            try:
                nodes = nodes_from_registry(await asyncio.to_thread(self._redis.hgetall, NODES_REDIS_KEY))
            except redis.RedisError:
                continue
            for node in self.heartbeats.update(nodes, time.time()):
                # A death is already being handled from Docker's events.
                if self.grouper.is_open(CONTAINER_PREFIX + node) or self._recently_reported(node):
                    continue
                asyncio.create_task(self._unresponsive(node, nodes[node]["heartbeat"]))

    async def _unresponsive(self, node: str, last_heartbeat: float) -> None:
        now = time.time()
        incident = Incident(kind="NODE_UNRESPONSIVE", node=node, first_event_at=last_heartbeat,
                            summary=f"Node '{node}' stopped responding: no heartbeat for {now - last_heartbeat:.0f}s. "
                                    "Checking what's wrong...",
                            timeline=[f"{clock(last_heartbeat)} last heartbeat", f"{clock(now)} heartbeat stale"])
        self._announce(incident)
        state = await inspect_container(CONTAINER_PREFIX + node)
        try:
            diagnosis = await asyncio.to_thread(self._get_json, "/diagnose", 60)
        except Exception as e:  # noqa: BLE001 -- still worth investigating without it
            diagnosis = {"error": str(e)}
        if not self.heartbeats.is_stale(node) and degraded_key(diagnosis) is None:
            self._close(incident, "dismissed", f"'{node}' started responding again on its own.")
            return
        self._reported_key = degraded_key(diagnosis) or self._reported_key
        incident.summary = (f"Node '{node}' stopped responding (no heartbeat since {clock(last_heartbeat)}), "
                            f"and {_container_condition(state)}.")
        incident.evidence = {"heartbeat_age_s": round(time.time() - last_heartbeat, 1), "container_state": state,
                             "diagnosis": compact_diagnosis(diagnosis) if "error" not in diagnosis else diagnosis}
        self._emit(incident)

    # services + pipeline sweep ----------------------------------------------------------

    async def _sweep_loop(self) -> None:
        while True:
            self._check_services()
            if "msight_api" not in self._down_reported and self._down_counts["msight_api"] == 0:
                await self._check_pipeline()
            # Sweep faster while something is open, so recovery shows up quickly.
            await asyncio.sleep(OPEN_SWEEP_SECONDS if self.open else self.sweep_seconds)

    def _check_services(self) -> None:
        for service, port in SERVICES.items():
            if _port_open(port):
                if service in self._down_reported:
                    for incident in [i for i in self.open.values() if i.node == service]:
                        self._close(incident, "resolved", f"{service} is listening on :{port} again.")
                self._down_counts[service] = 0
                self._down_reported.discard(service)
                continue
            self._down_counts[service] += 1
            if self._down_counts[service] == 2 and service not in self._down_reported:
                self._down_reported.add(service)
                now = time.time()
                self._emit(Incident(
                    kind="SERVICE_DOWN", node=service, first_event_at=now - self.sweep_seconds,
                    summary=f"Agent service '{service}' is not listening on :{port} (two checks in a row).",
                    timeline=[f"{clock(now)} :{port} refused twice"], evidence={"port": port}))

    def _resolve_recovered(self, diagnosis: dict) -> None:
        """Close incidents the verdict covers -- not ones whose facts are still settling
        (Docker may be mid-restart), nor service outages (closed by port). A stopped
        pipeline isn't a fixed one: those incidents are dismissed, not resolved."""
        if diagnosis.get("verdict") == "empty":
            stage, text = "dismissed", "The pipeline was stopped, so this was closed without being fixed."
        else:
            stage, text = "resolved", "The pipeline is healthy again: every node is running and data is flowing."
        for incident in list(self.open.values()):
            if (incident.kind != "SERVICE_DOWN" and incident.id in self._classified
                    and not self.grouper.is_open(CONTAINER_PREFIX + incident.node)):
                self._close(incident, stage, text)

    async def _check_pipeline(self) -> None:
        try:
            diagnosis = await asyncio.to_thread(self._get_json, "/diagnose", 60)
        except Exception as e:  # noqa: BLE001 -- a failed sweep is retried next time
            self.say(f"[watch] diagnose sweep failed: {e}")
            return
        if diagnosis.get("status") != "ok":  # no verdict at all: must not look healthy
            self.say(f"[watch] diagnose sweep failed: {diagnosis.get('message')}")
            return
        key = degraded_key(diagnosis)
        if key is None:
            self._resolve_recovered(diagnosis)
            self._pending_key, self._pending_count, self._reported_key = None, 0, None
            return

        self._pending_count = self._pending_count + 1 if key == self._pending_key else 1
        self._pending_key = key
        if self._pending_count < 2 or key == self._reported_key:
            return
        self._reported_key = key
        nodes = sorted({node for _, node in key})
        # A container/process/heartbeat incident for the same node already covers it.
        if all(self.grouper.is_open(CONTAINER_PREFIX + n) or self._recently_reported(n) for n in nodes):
            return
        now = time.time()
        self._emit(Incident(
            kind="PIPELINE_DEGRADED", node="+".join(nodes), first_event_at=now - self.sweep_seconds,
            summary=f"Pipeline degraded for two sweeps: {diagnosis.get('summary')}",
            timeline=[f"{clock(now - self.sweep_seconds)} first degraded sweep", f"{clock(now)} still degraded"],
            evidence={"diagnosis": compact_diagnosis(diagnosis)}))

    def _get_json(self, path: str, timeout: float) -> dict:
        with urllib.request.urlopen(self.api_url + path, timeout=timeout) as resp:
            return json.loads(resp.read())
