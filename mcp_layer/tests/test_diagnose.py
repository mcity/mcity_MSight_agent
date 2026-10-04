"""Graph diagnosis logic (msight_control_plane.diagnose_from_health) --
pure function over node_health() rows, no Docker/Redis needed."""
import pytest

from mcptools.msight_control_plane import diagnose_from_health, structural_findings


def node(name, health, pub=(), sub=(), in_rate=None, out_rate=None):
    return {"name": name, "health": health, "alive": health != "DEAD",
            "publish_topics": list(pub), "subscribe_topics": list(sub),
            "in_rate_hz": in_rate, "out_rate_hz": out_rate}


def src(h="OK"):
    return node("video_source", h, pub=["camera/s1"], out_rate=0 if h != "OK" else 60)


def det(h="OK"):
    return node("rfdetr_detector", h, pub=["detection/s1"], sub=["camera/s1"], in_rate=60, out_rate=0.2)


def view(h="OK"):
    return node("detection_viewer", h, sub=["detection/s1"], in_rate=0.2)


@pytest.mark.parametrize("rows, roots, affected", [
    pytest.param([src(), det(), view()], [], [], id="healthy"),
    pytest.param([src(), det("DEAD"), view("STARVED")],
                 ["rfdetr_detector:DEAD"], ["detection_viewer"], id="detector-dead"),
    pytest.param([src("DEAD"), det("DEAD"), view("STARVED")],
                 ["video_source:DEAD"], ["detection_viewer", "rfdetr_detector"],
                 id="source-dead-cascade"),
    pytest.param([src(), view("STARVED")],
                 ["detection_viewer:MISSING_PUBLISHER"], [], id="detector-deleted"),
    pytest.param([src(), det(), view(), node("viewer_2", "STARVED", sub=["detection/s_1"], in_rate=0)],
                 ["viewer_2:TOPIC_MISMATCH"], [], id="custom-node-topic-typo"),
    pytest.param([src(), det("STALLED"), view("STARVED")],
                 ["rfdetr_detector:STALLED"], ["detection_viewer"], id="detector-stalled"),
    pytest.param([node("lidar", "OK", pub=["pc/l1"], out_rate=10),
                  node("sort_tracker", "DEAD", pub=["tracks/l1"], sub=["pc/l1"]),
                  node("track_logger", "STARVED", sub=["tracks/l1"], in_rate=0)],
                 ["sort_tracker:DEAD"], ["track_logger"], id="custom-chain-no-demo-nodes"),
])
def test_root_causes_and_affected(rows, roots, affected):
    d = diagnose_from_health(rows)
    assert [f"{c['node']}:{c['kind']}" for c in d["root_causes"]] == roots
    assert sorted(a["node"] for a in d["affected"]) == sorted(affected)
    assert d["verdict"] == ("healthy" if not roots and not affected else "degraded")


def test_empty_pipeline():
    assert diagnose_from_health([])["verdict"] == "empty"


def test_suggested_fixes_never_name_tools():
    d = diagnose_from_health([src("DEAD"), det(), view("STARVED")])
    for c in d["root_causes"]:
        assert "_msight_" not in c["suggested_fix"]


def test_structural_findings_dead_publisher_is_not_also_missing():
    # A dead-but-registered publisher is reported as DEAD, not as a missing publisher.
    kinds = {f["kind"] for f in structural_findings([src(), det("DEAD"), view()])}
    assert kinds == {"DEAD"}
