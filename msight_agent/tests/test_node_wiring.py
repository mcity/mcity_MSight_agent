"""Node wiring checks found by launching every catalog node: data-kind mismatches crash a
node on its first message, and nodes added outside the pipeline outlive a stop."""
import asyncio
import json

from msight_agent.agent.prompts import STATE_HINTS
from msight_agent.msight import hints
from msight_agent.msight.runtime.executors import SupervisorExecutor, SupervisorHandle
from msight_agent.msight.runtime.node_catalog import NODE_CATALOG
from msight_agent.msight.tools import nodes, pipeline

KINDS = {"camera/s": ("image", "video_source", "video_source_rtsp"),
         "detection/s": ("detections", "rfdetr_detector", "rfdetr_detector")}


def test_viewer_on_a_camera_topic_is_refused_with_fitting_topics():
    problem = nodes._input_kind_problem(NODE_CATALOG["detection_viewer"], {"subscribe_topic": "camera/s"}, KINDS)
    assert "carries image" in problem and "['detection/s']" in problem


def test_matching_kind_and_unchecked_nodes_pass():
    assert nodes._input_kind_problem(NODE_CATALOG["detection_viewer"], {"subscribe_topic": "detection/s"}, KINDS) is None
    assert nodes._input_kind_problem(NODE_CATALOG["http_sink"], {"subscribe_topic": "camera/s"}, KINDS) is None


def test_removed_broken_nodes_are_not_offered():
    for broken in ("sdsm_encoder", "detection_results_viewer", "buffering_sort", "bytes_viewer"):
        assert broken not in NODE_CATALOG


def test_stop_reports_nodes_left_running(monkeypatch):
    class FakeCP:
        async def delete_node(self, name):
            pass

        async def live_node_names(self):
            return ["detection_viewer_2", "rfdetr_detector_2"]

    monkeypatch.setattr(pipeline, "get_control_plane", FakeCP)
    monkeypatch.setattr(pipeline, "get_msight_path", lambda: ("/m", None))
    fn = getattr(pipeline.stop_msight_pipeline, "fn", pipeline.stop_msight_pipeline)
    result = json.loads(asyncio.run(fn()))
    assert result["other_nodes_still_running"] == ["detection_viewer_2", "rfdetr_detector_2"]


def test_hint_mentions_extra_nodes_after_stop():
    live = {"pipeline": False, "extra_nodes": ["detection_viewer_2"]}
    assert "detection_viewer_2" in hints._extra_nodes_hint(live)
    assert "pipeline is stopped" in hints._extra_nodes_hint(live)
    assert hints._extra_nodes_hint({"pipeline": True, "extra_nodes": []}) == ""
    assert "msight_extra_nodes_running" in STATE_HINTS


def test_stopping_a_crashed_supervisor_node_still_cleans_up(tmp_path, monkeypatch):
    from msight_agent.msight.runtime import executors
    monkeypatch.setattr(executors, "NODE_LOG_DIR", tmp_path)

    async def run():
        proc = await asyncio.create_subprocess_exec("true")
        await proc.wait()  # the node "crashed"
        (tmp_path / "n.pid").write_text(str(proc.pid))
        await SupervisorExecutor().stop(SupervisorHandle("n", process=proc))
    asyncio.run(run())
    assert not (tmp_path / "n.pid").exists()


def test_remove_reports_what_is_still_running(monkeypatch):
    class FakeCP:
        def is_tracked(self, name):
            return True

        async def is_alive(self, name):
            return True

        async def delete_node(self, name):
            pass

        async def live_node_names(self):
            return ["img_test"]

    monkeypatch.setattr(nodes, "get_control_plane", FakeCP)
    monkeypatch.setattr(nodes, "get_msight_path", lambda: ("/m", None))
    fn = getattr(nodes.remove_msight_node, "fn", nodes.remove_msight_node)
    result = json.loads(asyncio.run(fn("folder_test")))
    assert result["still_running"] == ["img_test"] and "Still running: img_test" in result["message"]
