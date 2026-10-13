"""diagnose_from_health over synthetic node_health() rows (no Docker/Redis)."""
import pytest

from mcp_layer.mcptools.msight.diagnosis import diagnose_from_health, structural_findings


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


def removal(name, topics, seconds_ago=90):
    import time
    return {"at": time.time() - seconds_ago, "change": "removed", "node": name,
            "node_type": None, "publish_topics": topics}


def test_missing_publisher_names_the_removed_node():
    d = diagnose_from_health([src(), view("STARVED")],
                             [removal("rfdetr_detector", ["detection/s1"])])
    top = d["root_causes"][0]
    assert top["removed_publisher"] == "rfdetr_detector"
    assert "rfdetr_detector" in top["evidence"] and "rfdetr_detector" in top["suggested_fix"]


def test_unrelated_removal_is_not_blamed():
    d = diagnose_from_health([src(), view("STARVED")],
                             [removal("viewer_2", [])])
    assert "removed_publisher" not in d["root_causes"][0]


def _devices():
    return {
        "video_source": {"device": "cpu", "image": "michigantrafficlab/msight-vision:latest", "gpu_fallback": False},
        "rfdetr_detector": {"device": "cpu", "image": "msight-vision-cpu-local", "gpu_fallback": True},
        "gone": None,
    }


@pytest.mark.parametrize("gpu, expect", [
    pytest.param({"usable": False, "hardware": ["RTX 4090"], "reason": "nvidia-smi failed: no driver"},
                 "driver is the problem", id="hardware-but-broken-driver"),
    pytest.param({"usable": False, "hardware": [], "reason": "nvidia-smi is not installed"},
                 "not installed", id="no-gpu"),
    pytest.param({"usable": True, "hardware": ["RTX 4090"], "reason": None},
                 "restarting it would use the GPU", id="gpu-fixed-since-start"),
])
def test_compute_report_explains_cpu_fallback(monkeypatch, gpu, expect):
    import asyncio
    from mcp_layer.mcptools.msight import control_plane as cp_mod

    async def fake_gpu():
        return gpu

    monkeypatch.setattr(cp_mod, "gpu_report", fake_gpu)
    report = asyncio.run(cp_mod.MSightControlPlane().compute_report(_devices()))
    assert len(report["warnings"]) == 1
    assert "rfdetr_detector" in report["warnings"][0] and expect in report["warnings"][0]
    assert "gone" not in report["nodes"]


def test_compute_report_skips_gpu_probe_without_fallback(monkeypatch):
    # Status is polled by the dashboard; lspci/nvidia-smi only run when they explain something.
    import asyncio
    from mcp_layer.mcptools.msight import control_plane as cp_mod

    async def must_not_run():
        raise AssertionError("gpu_report should not be called")

    monkeypatch.setattr(cp_mod, "gpu_report", must_not_run)
    devices = {k: v for k, v in _devices().items() if k != "rfdetr_detector"}
    report = asyncio.run(cp_mod.MSightControlPlane().compute_report(devices))
    assert report["warnings"] == [] and "gpu" not in report
