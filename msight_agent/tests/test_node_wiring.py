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
    assert result["still_running"] == ["img_test"] and "added: img_test" in result["message"]


def test_nodes_are_listed_by_role():
    from msight_agent.msight.tools.pipeline import describe_by_role
    text = describe_by_role(["detection_viewer", "rfdetr_detector_2", "video_source", "local_dumper"])
    assert text == ("standard pipeline: detection_viewer, video_source; recording/archiving: local_dumper; "
                    "added: rfdetr_detector_2")


def test_wrong_topic_is_reported_before_missing_settings(monkeypatch):
    """The user picked a detection topic for a detector: say that, don't just ask for publish_topic."""
    class FakeCP:
        def get_status(self):
            return [{"name": "rfdetr_detector", "publish_topic": "detection/s"}]

        def started_publishers(self):
            return {}

        async def snapshot(self, names):
            return {n: {"alive": n == "rfdetr_detector", "device": None} for n in names}

        async def node_type_of(self, name):
            return "rfdetr_detector"

        async def is_alive(self, name):
            return True

    monkeypatch.setattr(nodes, "get_control_plane", FakeCP)
    monkeypatch.setattr(nodes, "get_msight_path", lambda: ("/m", None))
    fn = getattr(nodes.add_msight_node, "fn", nodes.add_msight_node)
    result = json.loads(asyncio.run(fn("rfdetr_detector", name="det2", config={"subscribe_topic": "detection/s"})))
    assert "carries detections" in result["message"] and "reads image" in result["message"]


def test_unknown_config_key_is_refused_with_valid_ones():
    from msight_agent.msight.runtime.node_catalog import unknown_config, valid_config_keys
    spec = NODE_CATALOG["video_source_mp4_folder"]
    assert unknown_config(spec, {"path": "/x", "publish_topic": "c", "sensor_name": "s"}) == ["path"]
    assert "folder" in valid_config_keys(spec)
    assert unknown_config(NODE_CATALOG["detection_viewer"], {"subscribe_topic": "d", "port": 9011}) == []


def test_every_internal_node_config_uses_valid_keys():
    """The pipeline start and recording build configs in code; they must not trip the new check."""
    from msight_agent.msight.runtime.node_catalog import unknown_config
    from msight_agent.msight.tools.pipeline import _fixed_pipeline, _resolve_video_source
    for source in (("/v.mp4", None), (None, "rtsp://cam")):
        for node in _fixed_pipeline(*_resolve_video_source(*source, "s"), "s"):
            assert unknown_config(NODE_CATALOG[node["node_type"]], node["config"]) == [], node
    recording = {"frame_annotator": {"subscribe_topic": "d", "publish_topic": "a"},
                 "video_aggregator": {"subscribe_topic": "a", "publish_topic": "v", "buffer_size": 40,
                                      "overlap_size": 0, "fps": 20},
                 "video_local_dumper": {"subscribe_topic": "v", "save_dir": "/x"},
                 "aws_video_pusher": {"subscribe_topic": "v", "bucket_name": "b", "prefix": ""}}
    for node_type, config in recording.items():
        assert unknown_config(NODE_CATALOG[node_type], config) == [], node_type


def test_topic_suggestions_only_offer_fitting_kinds():
    kinds = {"camera/s": ("image", "video_source", "video_source_rtsp"),
             "detection/s": ("detections", "rfdetr_detector", "rfdetr_detector")}
    text = nodes._describe_topics(kinds, NODE_CATALOG["rfdetr_detector"])
    assert "camera/s" in text and "detection/s" not in text
    assert "detection/s (detections)" in nodes._describe_topics(kinds, NODE_CATALOG["http_sink"])
    assert "add a node" in nodes._describe_topics({"detection/s": kinds["detection/s"]}, NODE_CATALOG["rfdetr_detector"])


def test_node_name_given_as_topic_resolves_to_its_topic():
    node_topics = {"video_source": ["camera/s"], "rfdetr_detector": ["detection/s"]}
    config, notes = nodes._resolve_node_names({"subscribe_topic": "video_source", "publish_topic": "d2"}, node_topics)
    assert config["subscribe_topic"] == "camera/s" and "is a node, not a topic" in notes[0]
    # A real topic is left alone, and so is an unknown name (the topic check reports it).
    assert nodes._resolve_node_names({"subscribe_topic": "camera/s"}, node_topics) == ({"subscribe_topic": "camera/s"}, [])
    assert nodes._resolve_node_names({"subscribe_topic": "nope"}, node_topics)[1] == []


def test_wiring_defaults_only_when_unambiguous():
    one_image = {"camera/s": ("image", "video_source", "video_source_rtsp"),
                 "detection/s": ("detections", "rfdetr_detector", "rfdetr_detector")}
    config, notes = nodes._fill_wiring_defaults(NODE_CATALOG["rfdetr_detector"], {}, "rfdetr_2", one_image)
    assert config == {"subscribe_topic": "camera/s", "publish_topic": "detection/rfdetr_2"} and len(notes) == 2
    two_images = {**one_image, "camera/t": ("image", "img", "local_image")}
    config, _ = nodes._fill_wiring_defaults(NODE_CATALOG["rfdetr_detector"], {}, "rfdetr_2", two_images)
    assert "subscribe_topic" not in config  # a real choice: left to the user
    config, notes = nodes._fill_wiring_defaults(NODE_CATALOG["detection_viewer"], {"subscribe_topic": "x"}, "v", one_image)
    assert config == {"subscribe_topic": "x"} and notes == []  # given values are never replaced
