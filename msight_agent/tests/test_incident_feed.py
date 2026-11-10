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


def test_multi_node_fix_message_restarts_nodes_without_consent_to_stop():
    from msight_agent.agent.guards import destructive_allowed
    fix = incidents.fix_action(_diagnosed("PIPELINE_DEGRADED", node="video_source+rfdetr_detector"))
    assert "video_source and rfdetr_detector" in fix["message"]
    assert destructive_allowed("restart_msight_node", {"name": "video_source"}, fix["message"], "")
    for tool, args in [("stop_msight_pipeline", {}), ("stop_msight_recording", {}),
                       ("remove_msight_node", {"name": "local_dumper"})]:
        assert not destructive_allowed(tool, args, fix["message"], ""), tool


def test_a_stopped_pipeline_dismisses_instead_of_resolving(quiet_watcher, monitor_events):
    w = quiet_watcher
    incident = Incident(kind="CONTAINER_KILLED", node="rfdetr_detector", summary="s", first_event_at=1)
    w.open[incident.id] = incident
    w._classified.add(incident.id)
    w._resolve_recovered({"verdict": "empty"})
    assert monitor_events == []                    # one empty sweep may be a restart in progress
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


# --- Redis outages and service checks --------------------------------------------------

import redis as redis_lib  # noqa: E402

from msight_agent.msight.routes import redis_outage_notice  # noqa: E402
from msight_agent.sentinel import stream as stream_mod  # noqa: E402
from msight_agent.sentinel.watcher import SERVICES  # noqa: E402


class FlakyRedis:
    """XADD fails while `down`, then records what it receives."""
    def __init__(self, *args, **kwargs):
        self.down, self.added = True, []

    def xadd(self, key, fields, **kwargs):
        if self.down:
            raise redis_lib.ConnectionError("Connection refused")
        self.added.append(json.loads(fields["data"])["stage"])


def test_events_held_during_a_redis_outage_are_published_in_order(monkeypatch):
    monkeypatch.setattr(stream_mod.redis, "Redis", FlakyRedis)
    publish = stream_mod.IncidentPublisher({}, say=lambda line: None)
    publish("INC-1", "detected")
    publish("INC-1", "classified")
    publish._client.down = False
    publish("INC-1", "resolved")
    assert publish._client.added == ["detected", "classified", "resolved"]


def test_a_long_outage_keeps_the_newest_events_and_says_how_many_were_dropped(monkeypatch):
    monkeypatch.setattr(stream_mod.redis, "Redis", FlakyRedis)
    monkeypatch.setattr(stream_mod, "HELD_EVENTS", 2)
    lines = []
    publish = stream_mod.IncidentPublisher({}, say=lines.append)
    for stage in ("detected", "classified", "investigating"):
        publish("INC-1", stage)
    publish._client.down = False
    publish("INC-1", "diagnosed")
    assert publish._client.added == ["investigating", "diagnosed"]
    assert "oldest 2 were dropped" in lines[-1]


def test_redis_outage_is_one_incident_and_resolves(quiet_watcher, monitor_events):
    w = quiet_watcher
    w._redis = redis_lib.Redis(host="localhost", port=6380)          # only its settings are read
    w._redis_unreachable(redis_lib.ConnectionError("refused"))
    assert monitor_events == []                                        # one failed read isn't an outage
    w._redis_unreachable(redis_lib.ConnectionError("refused"))
    w._redis_unreachable(redis_lib.ConnectionError("refused"))
    assert [a[1] for a, _ in monitor_events] == ["detected", "classified"]
    assert w.queue.get_nowait().node == "redis"
    w._redis_reachable()
    assert [a[1] for a, _ in monitor_events][-1] == "resolved"


def test_node_deaths_during_a_redis_outage_are_its_symptoms(quiet_watcher, monitor_events, monkeypatch):
    async def fake_inspect(name):
        return {"Running": False, "ExitCode": 1}

    monkeypatch.setattr(watcher_mod, "inspect_container", fake_inspect)
    w = quiet_watcher
    w._redis = redis_lib.Redis(host="localhost", port=6380)
    w._redis_unreachable(redis_lib.ConnectionError("refused"))
    w._redis_unreachable(redis_lib.ConnectionError("refused"))
    w.queue.get_nowait()                                                # the Redis incident itself
    w.grouper.add(DET, ev("die", 100.0, 1))
    asyncio.run(w._settle_once(103.0))
    assert len(monitor_events) == 2                                     # held: no announcement
    asyncio.run(w._settle_once(113.0))
    assert w.queue.empty()                                              # not investigated on its own
    assert "rfdetr_detector" in w._redis_incident.timeline[-1]


def test_the_chat_server_shows_its_own_redis_notice():
    notice = redis_outage_notice(1000.0)
    assert notice["status"] == "open" and notice["node"] == "redis" and incidents.fix_action(notice) is None
    assert "paused" in notice["note"]


def test_chat_server_is_watched():
    assert SERVICES["chat_server"] == 8001


def test_a_sweep_does_not_duplicate_an_open_incident(quiet_watcher, monitor_events, monkeypatch):
    # Live run: the detector exited and deregistered; the sweep saw two root causes
    # (detector down + viewer has no publisher) and raised a second card.
    w = quiet_watcher
    open_incident = Incident(kind="CONTAINER_EXITED", node="rfdetr_detector", summary="s", first_event_at=1)
    w.open[open_incident.id] = open_incident
    diagnosis = {"status": "ok", "verdict": "degraded", "summary": "detector down", "root_causes": [
        {"kind": "DEAD", "node": "rfdetr_detector"},
        {"kind": "MISSING_PUBLISHER", "node": "detection_viewer", "topic": "detection/s1"}]}
    monkeypatch.setattr(w, "_get_json", lambda path, timeout: diagnosis)
    asyncio.run(w._check_pipeline())
    asyncio.run(w._check_pipeline())                      # two bad sweeps in a row
    assert monitor_events == [] and w.queue.empty()


def test_a_new_fault_on_a_recently_fixed_node_is_reported(quiet_watcher, monkeypatch):
    # Live run: the detector was restarted (incident resolved), then frozen a minute later;
    # a 10-minute cooldown silently swallowed the freeze.
    w = quiet_watcher
    w._reported_at["rfdetr_detector"] = __import__("time").time() - 60   # fixed a minute ago
    assert not w._recently_reported("rfdetr_detector")
    still_open = Incident(kind="NODE_UNRESPONSIVE", node="rfdetr_detector", summary="s", first_event_at=1)
    w.open[still_open.id] = still_open
    assert w._recently_reported("rfdetr_detector")                       # but never twice while open


# --- robustness (code review) -----------------------------------------------------------

def test_two_incidents_for_one_node_in_one_second_get_distinct_ids():
    a = Incident(kind="CAMERA_BLACK", node="cam_src", summary="s", first_event_at=1, detected_at=1000.0)
    b = Incident(kind="CAMERA_COVERED", node="cam_src", summary="s", first_event_at=1, detected_at=1000.0)
    assert a.id != b.id and a.id == a.id and a.to_dict()["id"] == a.id


def test_a_failing_check_is_restarted_not_fatal(quiet_watcher, monkeypatch):
    lines, calls = [], []
    quiet_watcher.say = lines.append
    monkeypatch.setattr(watcher_mod.asyncio, "sleep", _no_sleep)

    async def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise redis_lib.ConnectionError("Redis went away mid-sample")

    asyncio.run(quiet_watcher._supervised("camera", flaky))
    assert len(calls) == 2 and "camera check failed" in lines[0]


async def _no_sleep(_seconds):
    return None


def test_frame_counts_during_a_redis_failure_are_discarded_not_zero(quiet_watcher):
    class BrokenPubSub:
        def subscribe(self, *topics):
            raise redis_lib.ConnectionError("down")

        def close(self):
            pass

    quiet_watcher._redis = type("R", (), {"pubsub": lambda self, **kw: BrokenPubSub()})()
    # Zeros would read as a frame-rate collapse and raise a false LOW_FPS alert.
    assert quiet_watcher._count_messages(["camera/x"], 0.01) is None
    assert quiet_watcher._one_message("camera/x", 0.01) is None


def test_reconnect_lists_the_notices_that_still_stand():
    from msight_agent.msight.routes import notices_sse
    assert notices_sse([]) == "event: notices\ndata: []\n\n"
    assert '"FEED-sentinel-5"' in notices_sse(["FEED-sentinel-5"])


def test_a_starved_node_is_reported_as_its_upstream(quiet_watcher, monitor_events, monkeypatch):
    # A stalled camera stream: video_source keeps heart-beating, the detector goes quiet.
    w = quiet_watcher
    diagnosis = {"status": "ok", "verdict": "degraded", "summary": "'video_source' published nothing.",
                 "root_causes": [{"kind": "STALLED", "node": "video_source"}],
                 "affected": [{"node": "rfdetr_detector", "health": "STARVED", "caused_by": "video_source"}]}

    async def fake_inspect(name):
        return {"Running": True, "Paused": False}
    monkeypatch.setattr(watcher_mod, "inspect_container", fake_inspect)
    monkeypatch.setattr(w, "_get_json", lambda path, timeout: diagnosis)
    monkeypatch.setattr(w.heartbeats, "is_stale", lambda node: True)

    asyncio.run(w._unresponsive("rfdetr_detector", last_heartbeat=1.0))
    emitted = w.queue.get_nowait()
    assert (emitted.kind, emitted.node) == ("PIPELINE_DEGRADED", "video_source")
    provisional = next(a[0] for a, k in monitor_events if a[1] == "detected" and k.get("node") == "rfdetr_detector")
    assert (provisional, "dismissed") in [(a[0], a[1]) for a, _ in monitor_events]


def _run_rate_loop(w, monkeypatch, samples):
    """Drive _rate_loop through scripted {topic: rate} samples, one per 30s tick."""
    script = iter(samples)
    monkeypatch.setattr(w, "_camera_sources", lambda: {
        "camera/c": {"source": "video_source", "detectors": {"rfdetr_detector": ["detection/c"]}}})
    monkeypatch.setattr(w, "_count_messages", lambda topics, window: next(script))
    w._redis = object()
    ticks = iter(range(len(samples)))

    async def tick(_seconds):
        if next(ticks, None) is None:
            raise asyncio.CancelledError
    monkeypatch.setattr(watcher_mod.asyncio, "sleep", tick)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(w._rate_loop())


def _stages(monitor_events):
    kinds = {a[0]: k.get("kind") for a, k in monitor_events if k.get("kind")}
    return [(kinds.get(a[0]), a[1]) for a, _ in monitor_events if a[1] in ("classified", "resolved")]


NORMAL = {"camera/c": 5.0, "detection/c": 5.0}


def test_no_slow_detector_alert_when_the_camera_itself_stopped(quiet_watcher, monitor_events, monkeypatch):
    stalled = {"camera/c": 0.0, "detection/c": 0.0}
    _run_rate_loop(quiet_watcher, monkeypatch, [NORMAL] * 4 + [stalled] * 3)
    kinds = [k for k, _ in _stages(monitor_events)]
    assert "CAMERA_SLOW_DETECTOR" not in kinds and "CAMERA_LOW_FPS" in kinds


def test_a_slow_detector_alert_clears_when_it_recovers(quiet_watcher, monitor_events, monkeypatch):
    slow = {"camera/c": 5.0, "detection/c": 0.0}
    _run_rate_loop(quiet_watcher, monkeypatch, [NORMAL] * 4 + [slow] * 2 + [NORMAL])
    assert _stages(monitor_events) == [("CAMERA_SLOW_DETECTOR", "classified"), ("CAMERA_SLOW_DETECTOR", "resolved")]


def test_a_paused_node_is_never_dismissed_as_recovered(quiet_watcher, monitor_events, monkeypatch):
    w = quiet_watcher

    async def paused(name):
        return {"Running": True, "Paused": True}
    monkeypatch.setattr(watcher_mod, "inspect_container", paused)
    monkeypatch.setattr(w, "_get_json", lambda path, timeout: {"status": "ok", "verdict": "healthy"})
    monkeypatch.setattr(w.heartbeats, "is_stale", lambda node: False)   # one read looked fine
    asyncio.run(w._unresponsive("rfdetr_detector", last_heartbeat=1.0))
    assert w.queue.get_nowait().node == "rfdetr_detector"
    assert "dismissed" not in [a[1] for a, _ in monitor_events]


def test_a_slow_but_working_node_is_not_reported_frozen(quiet_watcher, monitor_events, monkeypatch):
    # A slowed camera stream: the detector still works at 0.24 msg/s, so its heartbeat
    # (every 10 messages) goes quiet. That is a frame-rate drop, not a frozen node.
    w = quiet_watcher

    async def running(name):
        return {"Running": True, "Paused": False}
    monkeypatch.setattr(watcher_mod, "inspect_container", running)
    monkeypatch.setattr(w, "_get_json", lambda path, timeout: {"status": "ok", "verdict": "healthy", "nodes": [
        {"name": "rfdetr_detector", "health": "OK", "in_rate_hz": 0.24, "out_rate_hz": 0.24}]})
    monkeypatch.setattr(w.heartbeats, "is_stale", lambda node: True)
    asyncio.run(w._unresponsive("rfdetr_detector", last_heartbeat=1.0))
    assert w.queue.empty()
    assert [a[1] for a, _ in monitor_events] == ["detected", "dismissed"]
    # Not re-checked every ~40s, and not hidden from the frame-rate check either.
    assert w._slow_until["rfdetr_detector"] > 1.0 and not w._recently_reported("rfdetr_detector")


def test_a_new_camera_source_relearns_the_picture_and_closes_old_alerts(quiet_watcher, monitor_events, monkeypatch):
    from msight_agent.sentinel.camera_watch import CameraHealth
    w = quiet_watcher
    moved = Incident(kind="CAMERA_MOVED", node="video_source", summary="s", first_event_at=1)
    w.open[moved.id] = moved
    w._camera_health["camera/c"] = CameraHealth(source="rtsp://old/cam")
    w._camera_alerts[("camera/c", "MOVED")] = moved
    monkeypatch.setattr(watcher_mod.started_source, "source", lambda: "/videos/new.mp4")
    monkeypatch.setattr(watcher_mod, "CAMERA_REFERENCE_DIR", watcher_mod.LOG_DIR / "no_such_reference_dir")

    health = w._picture_health("camera/c")
    assert health.source == "/videos/new.mp4" and not health.learned
    assert (moved.id, "dismissed") in [(a[0], a[1]) for a, _ in monitor_events]
    assert w._camera_alerts == {}


def test_frame_rate_baselines_survive_a_restart_of_the_check(quiet_watcher, monitor_events, monkeypatch):
    _run_rate_loop(quiet_watcher, monkeypatch, [NORMAL] * 4)
    learned = quiet_watcher._rate_baselines["camera/c"].normal
    _run_rate_loop(quiet_watcher, monkeypatch, [NORMAL])     # the loop restarted after a failure
    assert learned == 5.0 and quiet_watcher._rate_baselines["camera/c"].normal == learned


def test_monitor_notice_opens_once_and_resolves_when_the_sentinel_is_back():
    from msight_agent.msight.routes import track_monitor
    notice, sent = track_monitor(100.0, None)
    assert notice["id"] == "FEED-sentinel-100" and '"status": "open"' in sent
    assert track_monitor(100.0, notice) == (notice, "")          # still down: nothing new to send
    notice, sent = track_monitor(None, notice)
    assert notice is None and '"status": "resolved"' in sent


def test_new_stream_entries_fold_into_state_and_advance_the_cursor():
    from msight_agent.msight.routes import new_events_sse
    states = {}
    entries = [(b"5-0", incidents.encode("INC-1", "detected", {"node": "x", "kind": "CONTAINER_KILLED"})),
               (b"6-0", {"data": "not json"}),
               (b"7-0", incidents.encode("INC-1", "resolved", {"text": "back"}))]
    chunk, cursor = new_events_sse(entries, states, "4-0")
    assert cursor == "7-0" and states["INC-1"]["status"] == "resolved"
    assert chunk.count("event: incident") == 2 and "id: 7-0" in chunk


def test_redis_notice_appears_only_after_a_few_seconds_down(monkeypatch):
    from msight_agent.msight import routes
    monkeypatch.setattr(routes.time, "time", lambda: 1000.0)
    since, notice, sent = routes.track_redis_outage(None, None)
    assert (since, notice, sent) == (1000.0, None, "")                     # a blip shows nothing
    monkeypatch.setattr(routes.time, "time", lambda: 1004.0)
    since, notice, sent = routes.track_redis_outage(since, None)
    assert notice["id"] == "FEED-redis-1000" and sent
    assert routes.track_redis_outage(since, notice)[2] == ""                # shown once


def test_an_incident_whose_start_fell_outside_the_window_is_never_open():
    # The window read begins after INC-1 was resolved, but includes the "fixed" event that
    # the autofix publishes just after the watcher closes it.
    window = [stage("INC-1", "fixed", 10, text="Fixed automatically"),
              stage("INC-2", "detected", 11, node="x", kind="CONTAINER_KILLED")]
    states = incidents.fold(window)
    assert states["INC-1"]["status"] == "unknown" and incidents.fix_action(states["INC-1"]) is None
    assert states["INC-2"]["status"] == "open"
    _, opened, _ = feed_start([("1-0", window[0]), ("2-0", window[1])])
    assert [s["id"] for s in opened] == ["INC-2"]


def test_a_late_event_of_an_incident_from_before_the_page_is_not_sent():
    from msight_agent.msight.routes import new_events_sse
    chunk, cursor = new_events_sse([(b"9-0", incidents.encode("INC-OLD", "fixed", {"text": "x"}))], {}, "8-0")
    assert chunk == "" and cursor == "9-0"
