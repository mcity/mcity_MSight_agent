"""Destructive tools (stop/remove/reset) only run when the user asked: the model once
stopped a running demo on its own to "recover" from an unrelated start error, and once
tried to stop it on "thanks, that's all for now"."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from msight_agent.agent.dispatch import ChatPipeline
from msight_agent.agent.guards import destructive_allowed
from msight_agent.agent.routing import Injection
from msight_agent.agent.state import MsightPipelineState, WorkflowState

STOP_OFFER = "The pipeline is still running. Would you like me to stop it?"


@pytest.mark.parametrize("tool, args, message, last_reply, allowed", [
    ("stop_msight_pipeline", {}, "stop the pipeline", "", True),
    ("stop_msight_pipeline", {}, "it's not working, stop it", "", True),
    ("stop_msight_pipeline", {}, "stop it and start the folder video instead", "", True),
    ("stop_msight_pipeline", {}, "don't stop it, just add a viewer", "", False),
    ("stop_msight_pipeline", {}, "add an mp4 folder source on /data/uploads, publishing camera/folder_test", "", False),
    ("stop_msight_pipeline", {}, "thanks, that's all for now", "", False),
    ("stop_msight_pipeline", {}, "yes", STOP_OFFER, True),
    ("stop_msight_pipeline", {}, "yes", "Shall I add a second viewer?", False),
    ("remove_msight_node", {"name": "viewer_2"}, "remove all the remaining nodes", "", True),
    ("remove_msight_node", {"name": "viewer_2"}, "get rid of the second viewer", "", True),
    ("remove_msight_node", {"name": "viewer_2"}, "yes", "Should I remove viewer_2?", True),
    ("remove_msight_node", {"name": "img_test"}, "yes", "Should I remove viewer_2?", False),  # other target
    ("remove_msight_node", {"name": "viewer_2"}, "the viewer looks frozen", "", False),
    ("stop_msight_recording", {}, "stop recording", "", True),
    ("stop_msight_recording", {}, "how long has it been recording?", "", False),
    ("reset_workflow_state", {}, "let's start over from scratch", "", True),
    ("reset_workflow_state", {}, "ok", "Do you want to discard all progress and start over?", True),
    ("reset_workflow_state", {}, "try again", "", False),
    ("add_msight_node", {}, "anything at all", "", True),  # not destructive
])
def test_destructive_needs_the_users_request(tool, args, message, last_reply, allowed):
    assert destructive_allowed(tool, args, message, last_reply) is allowed


def _dispatch(fn_name: str, fn_args: dict, message: str, last_reply: str = ""):
    p = ChatPipeline(mcp_client=MagicMock(), llm=MagicMock())
    p.state = WorkflowState(workflow_name="msight_pipeline",
                            msight_pipeline=MsightPipelineState(pipeline_running=True))
    p._user_texts = [message.lower()]
    p._last_assistant_text = last_reply
    client = MagicMock()
    client.call_tool = AsyncMock(return_value='{"status": "ok", "message": "done", "still_running": []}')
    messages: list = []
    with patch.object(WorkflowState, "save"):
        result, routings, _ = asyncio.run(p._dispatch(
            fn_name, fn_args, SimpleNamespace(id="c1"), client, messages))
    return client.call_tool.called, result, routings, messages


def test_blocked_call_never_reaches_the_tool_and_asks_first():
    called, result, routings, messages = _dispatch("remove_msight_node", {"name": "viewer_2"}, "thanks!")
    assert not called
    assert json.loads(result)["status"] == "error"
    assert isinstance(routings[0], Injection) and "Should I remove viewer_2?" in routings[0].message
    assert messages[-1]["tool_call_id"] == "c1"  # every tool call still gets its result


def test_requested_call_goes_through():
    called, *_ = _dispatch("stop_msight_pipeline", {}, "yes", STOP_OFFER)
    assert called
