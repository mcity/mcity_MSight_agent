"""add_msight_node refuses guessed topics and taken ports, and finds a free port for viewers."""
import pytest

from msight_agent.msight.runtime.control_plane import web_port_from_cmd
from msight_agent.msight.runtime.node_catalog import NODE_CATALOG
from msight_agent.msight.tools import nodes

PUBLISHED = {"camera/gs_mcity_1", "detection/gs_mcity_1"}


def test_guessed_topic_is_refused_with_the_real_ones():
    problem = nodes._input_topic_problem({"subscribe_topic": "detection_results"}, PUBLISHED)
    assert "detection_results" in problem and "detection/gs_mcity_1" in problem


def test_published_topic_passes():
    assert nodes._input_topic_problem({"subscribe_topic": "detection/gs_mcity_1"}, PUBLISHED) is None


def test_empty_pipeline_allows_any_topic():
    # Nodes may be added in any order, before their publisher exists.
    assert nodes._input_topic_problem({"subscribe_topic": "anything"}, set()) is None


@pytest.fixture
def busy_ports(monkeypatch):
    taken = {9010, 9011}
    monkeypatch.setattr(nodes, "port_in_use", lambda p: p in taken)
    monkeypatch.setattr(nodes, "next_free_port", lambda start: next(p for p in range(start, start + 50) if p not in taken))
    return taken


def test_viewer_without_port_gets_a_free_one(busy_ports):
    config, problem = nodes._resolve_web_port(NODE_CATALOG["detection_viewer"], {"subscribe_topic": "x"})
    assert problem is None and config["port"] == 9012


def test_taken_port_is_refused(busy_ports):
    _, problem = nodes._resolve_web_port(NODE_CATALOG["detection_viewer"], {"port": 9010})
    assert "9010" in problem and "9012" in problem


def test_non_web_node_is_untouched(busy_ports):
    config, problem = nodes._resolve_web_port(NODE_CATALOG["rfdetr_detector"], {"det_configs": "x"})
    assert problem is None and "port" not in config


@pytest.mark.parametrize("cmd, port", [
    ("/bin/bash -c exec msight_launch_web_viewer --name v --subscribe-topic t --port 9011", 9011),
    ("/bin/bash -c exec msight_launch_web_viewer --name v --subscribe-topic t", 9010),
    ("/bin/bash -c exec msight_launch_rfdetr_detection --name d", None),
])
def test_web_port_from_cmd(cmd, port):
    assert web_port_from_cmd(cmd) == port
