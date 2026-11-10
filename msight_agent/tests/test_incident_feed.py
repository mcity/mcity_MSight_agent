"""The live incident feed: early announcement, heartbeats, folding stream events into
incident states, the chat's SSE start, the Fix-it action and the chat agent's tool.
No Docker, Redis or LLM: Redis is faked, Docker calls are patched."""
import asyncio
import json

import pytest

from msight_agent.msight.routes import feed_start
from msight_agent.msight.runtime import incidents
from msight_agent.msight.tools import incidents as incident_tool
from msight_agent.sentinel import watcher as watcher_mod
from msight_agent.sentinel.stream import friendly_step
from msight_agent.sentinel.watcher import EventGrouper, HeartbeatWatch, Incident, Watcher, describe_failure

DET = "msight-rfdetr_detector"


def ev(action, at, exit_code=None, signal=None):
    return {"at": at, "action": action, "exit_code": exit_code, "signal": signal}


def stage(incident_id, name, at, **fields):
    return {"incident": incident_id, "stage": name, "at": at, **fields}


REPORT = {"summary": "s", "root_cause": "SIGKILL via Docker", "category": "external_action",
          "confidence": "high", "evidence": ["e1"], "affected": ["rfdetr_detector"],
          "suggested_fix": "restart it", "prevention": "p"}


# --- announcing early -----------------------------------------------------------------

def test_a_death_is_announced_after_the_grace_period_not_after_settling():
    grouper = EventGrouper(settle=12)
    grouper.add(DET, ev("kill", 100.0, signal="SIGKILL"))
    grouper.add(DET, ev("die", 100.1, 137))
    assert grouper.announceable(now=101.0, grace=2) == []
    assert [n for n, _ in grouper.announceable(now=102.2, grace=2)] == [DET]
    assert grouper.settled(now=102.2) == []                       # facts still settling


def test_a_deliberate_stop_is_never_announced():
    grouper = EventGrouper(settle=12)
    grouper.add(DET, ev("kill", 100.0, signal="SIGTERM"))
    grouper.add(DET, ev("die", 100.4, 143))
    grouper.add(DET, ev("destroy", 100.5))                       # the control plane's docker rm
    assert grouper.announceable(now=110, grace=2) == []


def test_provisional_description_needs_no_container_state():
    assert describe_failure([ev("oom", 1), ev("die", 1, 137)], {})[0] == "CONTAINER_OOM"
    assert describe_failure([ev("kill", 1, signal="SIGKILL"), ev("die", 1, 137)], {})[0] == "CONTAINER_KILLED"


@pytest.fixture
def quiet_watcher(monitor_events, monkeypatch):
    monkeypatch.setattr(watcher_mod, "append_event", lambda *a, **k: None)
    w = Watcher(asyncio.Queue(), say=lambda line: None, publish=lambda *a, **k: monitor_events.append((a, k)))
    return w


@pytest.fixture
def monitor_events():
    return []


def test_watcher_announces_then_classifies_the_same_incident(quiet_watcher, monitor_events, monkeypatch):
    async def fake_inspect(name):
        return {"Running": False, "OOMKilled": False, "ExitCode": 137}

    async def fake_run(*argv, timeout=15):
        return 0, "last log line"

    monkeypatch.setattr(watcher_mod, "inspect_container", fake_inspect)
    monkeypatch.setattr(watcher_mod, "_run", fake_run)
    w = quiet_watcher
    w.grouper.add(DET, ev("kill", 100.0, signal="SIGKILL"))
    w.grouper.add(DET, ev("die", 100.0, 137))

    asyncio.run(w._settle_once(102.5))                           # announced, not yet settled
    assert [a[1] for a, _ in monitor_events] == ["detected"]
    assert w.queue.empty()

    asyncio.run(w._settle_once(113.0))                           # quiet for 12s: classified + queued
    stages = [a[1] for a, _ in monitor_events]
    assert stages == ["detected", "classified"]
    assert len({a[0] for a, _ in monitor_events}) == 1           # one incident id throughout
    assert w.queue.get_nowait().kind == "CONTAINER_KILLED"


def test_recovery_closes_only_settled_incidents(quiet_watcher, monitor_events):
    w = quiet_watcher
    settled = Incident(kind="CONTAINER_KILLED", node="rfdetr_detector", summary="s", first_event_at=1)
    settling = Incident(kind="CONTAINER_OOM", node="video_source", summary="s", first_event_at=1)
    outage = Incident(kind="SERVICE_DOWN", node="msight_api", summary="s", first_event_at=1)
    for i in (settled, settling, outage):
        w.open[i.id] = i
    w._classified |= {settled.id, outage.id}                     # 'settling' is still provisional
    w._resolve_recovered({"verdict": "healthy"})
    assert set(w.open) == {settling.id, outage.id}
    assert [(a[0], a[1]) for a, _ in monitor_events] == [(settled.id, "resolved")]


# --- heartbeats -----------------------------------------------------------------------

def chain(src_hb, det_hb, view_hb):
    return {"video_source": {"heartbeat": src_hb, "publish": ["camera"], "subscribe": []},
            "rfdetr_detector": {"heartbeat": det_hb, "publish": ["detection"], "subscribe": ["camera"]},
            "detection_viewer": {"heartbeat": view_hb, "publish": [], "subscribe": ["detection"]}}


def learn(watch, start=0.0, seconds=10, gap=1.0):
    """Feed a healthy pipeline that heartbeats every `gap` seconds; returns the last time."""
    t = start
    while t < start + seconds:
        watch.update(chain(t, t, t), now=t)
        t += gap
    return t - gap


def test_a_frozen_node_is_flagged_once():
    watch = HeartbeatWatch(stale_after=5)
    t = learn(watch)
    # The detector freezes; the viewer, fed by it, goes quiet too.
    assert watch.update(chain(t + 3, t, t), now=t + 3) == []
    assert watch.update(chain(t + 6, t, t), now=t + 6) == ["rfdetr_detector"]
    assert watch.update(chain(t + 7, t, t), now=t + 7) == []          # reported once


def test_a_node_starved_by_a_dead_upstream_is_not_blamed():
    # The live false positive: the detector was killed and the viewer stopped heart-beating.
    watch = HeartbeatWatch(stale_after=5)
    t = learn(watch)
    flagged = watch.update(chain(t + 8, t, t), now=t + 8)
    assert "detection_viewer" not in flagged


def test_a_node_is_not_blamed_while_its_restarted_upstream_comes_back():
    # The live race: Fix it recreated the detector; it re-registered (fresh heartbeat)
    # before its model loaded, while the viewer was still waiting for detections.
    watch = HeartbeatWatch(stale_after=5)
    t = learn(watch)
    gone = chain(t + 8, t, t)
    del gone["rfdetr_detector"]
    assert watch.update(gone, now=t + 8) == []
    assert watch.update(chain(t + 9, t + 9, t), now=t + 9) == []        # re-registered, model loading
    assert watch.update(chain(t + 20, t + 9, t), now=t + 20) == []      # still loading: not proven yet
    assert watch.update(chain(t + 21, t + 21, t), now=t + 21) == []     # heartbeat advanced: working
    assert watch.update(chain(t + 24, t + 24, t), now=t + 24) == []     # viewer gets its threshold
    assert watch.update(chain(t + 27, t + 27, t), now=t + 27) == ["detection_viewer"]  # still stuck: real


def test_a_node_whose_publisher_was_removed_is_a_symptom():
    watch = HeartbeatWatch(stale_after=5)
    t = learn(watch)
    nodes = chain(t + 8, t + 8, t)
    del nodes["rfdetr_detector"]                                       # deregistered
    assert watch.update(nodes, now=t + 8) == []


def test_a_slow_node_gets_a_learned_threshold():
    # RF-DETR on CPU heart-beats every ~45s (every 10 frames at ~0.2 fps).
    watch = HeartbeatWatch(stale_after=5)
    t = learn(watch, seconds=200, gap=45)
    assert watch.threshold("rfdetr_detector") == 135
    assert watch.update(chain(t + 60, t, t), now=t + 60) == []         # normal for this node


def test_ghosts_at_startup_are_ignored_and_recovery_is_forgotten():
    watch = HeartbeatWatch(stale_after=5, unlearned=30)
    assert watch.update(chain(100, 0, 100), now=100) == []             # first look only learns
    assert watch.update(chain(101, 0, 101), now=101) == []             # ghost stays quiet
    assert not watch.is_stale("detection_viewer")


# --- folding the stream ---------------------------------------------------------------

def test_fold_builds_one_state_per_incident():
    states = incidents.fold([
        stage("A", "detected", 1, node="rfdetr_detector", kind="CONTAINER_KILLED", summary="killed"),
        stage("A", "investigating", 2),
        *[stage("A", "progress", 3 + i, text=f"step {i}") for i in range(15)],
        stage("A", "diagnosed", 20, report=REPORT, checks=[]),
    ])
    a = states["A"]
    assert (a["status"], a["stage"], a["node"], a["detected_at"]) == ("open", "diagnosed", "rfdetr_detector", 1)
    assert a["steps"][-1] == "step 14" and len(a["steps"]) == 12
    assert a["report"]["root_cause"] == "SIGKILL via Docker"


def test_a_diagnosis_after_recovery_keeps_the_incident_resolved():
    a = incidents.fold([stage("A", "detected", 1), stage("A", "resolved", 5, text="healthy again"),
                        stage("A", "diagnosed", 9, report=REPORT)])["A"]
    assert a["status"] == "resolved" and a["closed_reason"] == "healthy again" and a["report"]


# --- the Fix-it action ----------------------------------------------------------------

def _diagnosed(kind, node="rfdetr_detector"):
    return incidents.fold([stage("INC-1", "detected", 1, node=node, kind=kind),
                           stage("INC-1", "diagnosed", 2, report=REPORT)])["INC-1"]


@pytest.mark.parametrize("kind", ["CONTAINER_KILLED", "CONTAINER_OOM", "NODE_UNRESPONSIVE"])
def test_fix_message_is_consent_for_a_targeted_restart(kind):
    from msight_agent.agent.guards import CHANGE_REQUEST_RE, destructive_allowed
    fix = incidents.fix_action(_diagnosed(kind))
    assert fix["label"] == "Fix it: restart rfdetr_detector"
    assert CHANGE_REQUEST_RE.search(fix["message"])
    # The click is the user's request for exactly this restart -- and nothing broader.
    assert destructive_allowed("restart_msight_node", {"name": "rfdetr_detector"}, fix["message"], "")
    assert not destructive_allowed("stop_msight_pipeline", {}, fix["message"], "")


def test_a_stopped_pipeline_dismisses_instead_of_resolving(quiet_watcher, monitor_events):
    w = quiet_watcher
    incident = Incident(kind="CONTAINER_KILLED", node="rfdetr_detector", summary="s", first_event_at=1)
    w.open[incident.id] = incident
    w._classified.add(incident.id)
    w._resolve_recovered({"verdict": "empty"})
    assert [a[1] for a, _ in monitor_events] == ["dismissed"]


def test_no_fix_button_before_a_diagnosis_or_for_service_outages():
    undiagnosed = incidents.fold([stage("INC-1", "detected", 1, node="x", kind="CONTAINER_KILLED")])["INC-1"]
    assert incidents.fix_action(undiagnosed) is None
    assert incidents.fix_action(_diagnosed("SERVICE_DOWN", node="msight_api")) is None


# --- the chat's SSE start -------------------------------------------------------------

RECENT = [("1-0", stage("A", "detected", 1, node="n", kind="CONTAINER_KILLED")),
          ("2-0", stage("B", "detected", 2, node="m", kind="CONTAINER_OOM")),
          ("3-0", stage("B", "resolved", 3, text="ok")),
          ("4-0", stage("A", "progress", 4, text="reading logs"))]


def test_a_fresh_page_gets_every_open_incident():
    states, opened, cursor = feed_start(RECENT)
    assert [s["id"] for s in opened] == ["A"] and cursor == "4-0"


def test_a_reconnect_resumes_without_replaying_steps_twice():
    states, opened, cursor = feed_start(RECENT, last_event_id="3-0")
    assert opened == [] and cursor == "3-0"
    assert states["A"]["steps"] == []          # "4-0" will arrive via XREAD and be applied once


# --- the chat agent's tool ------------------------------------------------------------

class FakeRedis:
    def __init__(self, *args, **kwargs):
        pass

    def xrevrange(self, key, count=500):
        now = __import__("time").time()
        events = [stage("INC-1", "detected", now - 30, node="rfdetr_detector", kind="CONTAINER_KILLED",
                        summary="killed via Docker"),
                  stage("INC-1", "diagnosed", now - 5, report=REPORT, checks=["a check"])]
        return [(f"{i}-0".encode(), {b"data": json.dumps(e).encode()}) for i, e in reversed(list(enumerate(events)))]


def test_get_recent_incidents_gives_the_report_and_fix(monkeypatch):
    monkeypatch.setattr(incident_tool.redis, "Redis", FakeRedis)
    result = json.loads(asyncio.run(incident_tool.get_recent_incidents()))
    [inc] = result["incidents"]
    assert inc["diagnosis"]["root_cause"] == "SIGKILL via Docker"
    assert inc["checks"] == ["a check"] and inc["fix"]["label"] == "Fix it: restart rfdetr_detector"
    assert "restart_msight_node" in result["fix_guidance"]


def test_get_recent_incidents_unknown_id(monkeypatch):
    monkeypatch.setattr(incident_tool.redis, "Redis", FakeRedis)
    assert json.loads(asyncio.run(incident_tool.get_recent_incidents("INC-nope")))["status"] == "error"


# --- progress wording -----------------------------------------------------------------

@pytest.mark.parametrize("tool, args, expected", [
    ("mcp__msight__diagnose_msight_pipeline", {}, "Measuring live data flow"),
    ("mcp__msight__inspect_host", {"command": "docker inspect msight-rfdetr_detector"}, "Running `docker inspect"),
    ("Read", {"file_path": "/repo/msight_agent/msight/runtime/executors.py"}, "Reading executors.py"),
])
def test_friendly_steps(tool, args, expected):
    assert friendly_step(tool, args).startswith(expected)


def test_incidents_left_open_by_a_previous_run_are_closed(monitor_events, monkeypatch):
    monkeypatch.setattr(watcher_mod, "recent_incidents", lambda client: [{"id": "INC-old"}])
    w = Watcher(asyncio.Queue(), say=lambda line: None, publish=lambda *a, **k: monitor_events.append((a, k)))
    w._redis = object()                                           # anything non-None; the read is patched
    w.close_orphans()
    assert [(a[0], a[1]) for a, _ in monitor_events] == [("INC-old", "dismissed")]
