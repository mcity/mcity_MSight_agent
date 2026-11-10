"""The autonomous fixer's guardrails: what it may touch, when it may act at all, and how
its outcome shows up in the incident feed. No LLM, Docker or Redis."""
import pytest

from msight_agent.msight.runtime import incidents
from msight_agent.sentinel.checks import autofix_decision
from msight_agent.sentinel.policy import FixPolicy, incident_nodes

MCP = "mcp__msight__"
REPORT = {"summary": "s", "root_cause": "killed via docker", "category": "external_action",
          "confidence": "high", "evidence": ["e"], "affected": ["rfdetr_detector"],
          "suggested_fix": "restart it", "prevention": "p", "requires_human": False}


# --- scope ----------------------------------------------------------------------------

def test_fixer_may_restart_only_the_incidents_node():
    policy = FixPolicy(["rfdetr_detector"])
    assert policy(MCP + "restart_msight_node", {"name": "rfdetr_detector"}) is None
    assert "only restart" in policy(MCP + "restart_msight_node", {"name": "video_source"})


@pytest.mark.parametrize("tool, args", [
    ("stop_msight_pipeline", {}),
    ("remove_msight_node", {"name": "rfdetr_detector"}),
    ("add_msight_node", {"node_type": "rfdetr_detector"}),
    ("start_msight_archiving", {"s3_bucket": "attacker-bucket"}),   # would redirect footage
])
def test_fixer_cannot_stop_remove_add_or_archive(tool, args):
    assert FixPolicy(["rfdetr_detector"])(MCP + tool, args) is not None


def test_fixer_cannot_recompose_the_pipeline_or_change_the_source():
    # start_msight_pipeline recomposes: it would stop a running recording or user-added nodes.
    assert FixPolicy(["video_source"])(MCP + "start_msight_pipeline", {}) is not None
    recording = FixPolicy(["local_dumper"])
    assert "current settings" in recording(MCP + "start_msight_recording", {"sensor_name": "other_cam"})
    assert recording(MCP + "start_msight_recording", {}) is None


def test_fixer_budget_covers_each_node_of_a_multi_node_incident():
    policy = FixPolicy(["video_source", "rfdetr_detector", "detection_viewer"])
    for node in ("video_source", "rfdetr_detector", "detection_viewer"):
        assert policy(MCP + "restart_msight_node", {"name": node}) is None
    assert policy.budget == 6


def test_fixer_has_an_action_budget():
    policy = FixPolicy(["rfdetr_detector"])
    assert policy(MCP + "restart_msight_node", {"name": "rfdetr_detector"}) is None
    assert policy(MCP + "restart_msight_node", {"name": "rfdetr_detector"}) is None
    assert "Only 2" in policy(MCP + "restart_msight_node", {"name": "rfdetr_detector"})
    assert len(policy.actions) == 2


def test_fixer_keeps_the_investigators_read_rules():
    policy = FixPolicy(["rfdetr_detector"])
    assert policy(MCP + "diagnose_msight_pipeline", {}) is None
    assert policy("Bash", {"command": "rm -rf /"}) is not None


def test_recording_restart_only_for_recording_incidents():
    assert FixPolicy(["local_dumper"])(MCP + "start_msight_recording", {}) is None
    assert FixPolicy(["rfdetr_detector"])(MCP + "start_msight_recording", {}) is not None


def test_incident_nodes_split():
    assert incident_nodes("detection_viewer+rfdetr_detector") == ["detection_viewer", "rfdetr_detector"]


# --- when it may act ------------------------------------------------------------------

INCIDENT = {"kind": "CONTAINER_KILLED", "node": "rfdetr_detector"}


@pytest.mark.parametrize("report, warnings, incident, past, expect_fix, why", [
    (REPORT, [], INCIDENT, [], True, ""),
    (None, [], INCIDENT, [], False, "no diagnosis"),
    ({**REPORT, "requires_human": True, "human_action": "Re-aim the camera."}, [], INCIDENT, [], False, "Re-aim"),
    (REPORT, ["Destructive advice in suggested_fix: 'rm -rf'"], INCIDENT, [], False, "consistency checks"),
    ({**REPORT, "confidence": "low"}, [], INCIDENT, [], False, "low confidence"),
    (REPORT, [], {"kind": "SERVICE_DOWN", "node": "msight_api"}, [], False, "can't be restarted"),
    (REPORT, [], INCIDENT, [990.0, 995.0], False, "keeps failing"),          # flapping
    (REPORT, [], INCIDENT, [-5000.0, -4000.0], True, ""),                    # fixes over 30 min ago don't count
])
def test_autofix_decision(report, warnings, incident, past, expect_fix, why):
    ok, reason = autofix_decision(incident, report, warnings, past, now=1000.0)
    assert ok is expect_fix and why in reason


# --- feed states and the Fix-it button ------------------------------------------------

def ev(stage, at, **fields):
    return {"incident": "INC-1", "stage": stage, "at": at, **fields}


def state(*events):
    return incidents.fold([ev("detected", 0, node="rfdetr_detector", kind="CONTAINER_KILLED"), *events])["INC-1"]


def test_autofix_flow_in_the_feed():
    s = state(ev("diagnosed", 1, report=REPORT, autonomy="fix"), ev("fixing", 2))
    assert s["stage"] == "fixing" and incidents.fix_action(s) is None        # no button while Claude fixes
    s = state(ev("diagnosed", 1, report=REPORT, autonomy="fix"), ev("fixing", 2),
              ev("resolved", 3, text="healthy"), ev("fixed", 4, text="Fixed automatically: restart"))
    assert s["status"] == "resolved" and s["fix_outcome"] == "fixed" and "restart" in s["fix_text"]


def test_escalation_keeps_the_button_for_a_person_unless_a_person_is_needed():
    s = state(ev("diagnosed", 1, report=REPORT, autonomy="fix"), ev("escalated", 2, text="didn't recover"))
    assert s["stage"] == "escalated" and incidents.fix_action(s) is not None
    s = state(ev("diagnosed", 1, report={**REPORT, "requires_human": True}, autonomy="suggest"))
    assert incidents.fix_action(s) is None


def test_observe_mode_offers_no_fix():
    assert incidents.fix_action(state(ev("diagnosed", 1, report=REPORT, autonomy="observe"))) is None


def test_restarting_a_recording_node_points_to_start_recording():
    import asyncio
    import json
    from msight_agent.msight.tools.pipeline import restart_msight_node
    result = json.loads(asyncio.run(restart_msight_node("local_dumper")))
    assert "start_msight_recording" in json.dumps(result)


def test_docker_start_waits_out_a_name_still_in_use(monkeypatch, tmp_path):
    import asyncio
    from msight_agent.msight.runtime import executors
    calls = []

    async def fake_run_cmd(*argv, **kwargs):
        calls.append(argv[:2])
        if argv[:2] == ("docker", "run") and sum(c == ("docker", "run") for c in calls) == 1:
            return 125, 'docker: Error response from daemon: Conflict. The container name "/msight-x" is already in use'
        return 0, "abc123"

    async def no_sleep(_):
        return None
    monkeypatch.setattr(executors, "run_cmd", fake_run_cmd)
    monkeypatch.setattr(executors.asyncio, "sleep", no_sleep)
    handle = asyncio.run(executors.DockerExecutor().start("x", ["true"], tmp_path, {}))
    assert handle.container_name.endswith("x") and calls.count(("docker", "run")) == 2
