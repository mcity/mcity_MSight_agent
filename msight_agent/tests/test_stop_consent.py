"""stop_msight_pipeline only runs when the user asked; the model once stopped a running
demo on its own to "recover" from an unrelated start error."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from msight_agent.agent.dispatch import ChatPipeline
from msight_agent.agent.routing import Injection
from msight_agent.agent.state import MsightPipelineState, WorkflowState


def _stop(current: str, last_reply: str = ""):
    p = ChatPipeline(mcp_client=MagicMock(), llm=MagicMock())
    p.state = WorkflowState(workflow_name="msight_pipeline",
                            msight_pipeline=MsightPipelineState(pipeline_running=True))
    p._user_texts = [current.lower()]
    p._last_assistant_text = last_reply
    client = MagicMock()
    client.call_tool = AsyncMock(return_value='{"status": "ok", "message": "stopped"}')
    with patch.object(WorkflowState, "save"):
        result, routings = asyncio.run(p._handle_stop_msight_pipeline({}, client))
    return client.call_tool.called, routings


@pytest.mark.parametrize("current, last_reply, allowed", [
    ("add an mp4 folder source on /data/uploads, publishing camera/folder_test", "", False),
    ("stop the pipeline", "", True),
    ("stop it and start the folder video instead", "", True),
    ("yes", "The pipeline is still running. Would you like me to stop it?", True),
    ("yes", "Shall I add a second viewer?", False),
])
def test_stop_needs_the_users_request(current, last_reply, allowed):
    called, routings = _stop(current, last_reply)
    assert called is allowed
    if not allowed:
        assert isinstance(routings[0], Injection)
