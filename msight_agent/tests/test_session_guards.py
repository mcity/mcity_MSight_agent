"""Guards against the model acting on things the user didn't say: a stale session's
history, a tool instruction left in a reply, an invented choice, or consent for a
summary that was never shown. WorkflowState.save is patched so config.py is untouched."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from msight_agent.agent.dispatch import ChatPipeline
from msight_agent.agent.routing import HardStop, Injection, Sentinels
from msight_agent.agent.state import AutoLabelingPhase, AutoLabelingState, WorkflowState
from msight_agent.auto_labeling.handlers import _BACKEND_USER_TOKENS, _value_named_by_user
from msight_agent.chat_server import MAX_HISTORY_TURNS, history_for_session


@pytest.fixture(autouse=True)
def no_save():
    with patch.object(WorkflowState, "save"):
        yield


def _pipeline(state: WorkflowState, user_messages: list[str] = ()) -> ChatPipeline:
    p = ChatPipeline(mcp_client=MagicMock(), llm=MagicMock())
    p.state = state
    p._user_texts = [m.lower() for m in user_messages]
    return p


def _client():
    client = MagicMock()
    client.call_tool = AsyncMock(return_value="ok")
    return client


@pytest.mark.parametrize("phase, action", [
    (AutoLabelingPhase.TRAINING, "auto-labeling run"),
    (AutoLabelingPhase.ANNOTATING, "annotation export"),
])
def test_locked_reply_names_no_tool_and_asks_no_yes_no(phase, action):
    state = WorkflowState(workflow_name="auto_labeling", dataset_name="ds", dataset_confirmed=True,
                          auto_labeling=AutoLabelingState(phase=phase))
    result, routings = asyncio.run(_pipeline(state)._handle_select_or_switch_workflow(
        "switch_workflow", {"workflow_name": "auto_labeling"}, _client(), []))
    reply = routings[0].reply
    assert isinstance(routings[0], HardStop)
    assert "switch_workflow" not in reply and "(" not in reply and "?" not in reply
    assert action in reply
    assert "confirm_restart=true" in result      # the model still gets the instruction


def test_history_trimmed_to_current_session():
    history = [[f"u{i}", f"a{i}"] for i in range(6)]
    assert history_for_session(history, WorkflowState(turns_since_reset=0)) == []
    assert history_for_session(history, WorkflowState(turns_since_reset=2)) == history[-2:]
    assert history_for_session(history, WorkflowState(turns_since_reset=99)) == history[-MAX_HISTORY_TURNS:]
    assert history_for_session(history, None) == history[-MAX_HISTORY_TURNS:]


def test_switch_keeps_history_budget():
    state = WorkflowState(workflow_name="auto_labeling", turns_since_reset=3)
    assert state.reset_for_workflow("auto_labeling").turns_since_reset == 3


def test_backend_must_be_named_by_user():
    assert _value_named_by_user("cvat", _BACKEND_USER_TOKENS, ["use cvat please"])
    assert not _value_named_by_user("cvat", _BACKEND_USER_TOKENS, ["try again"])
    # An earlier comparison question is not a selection.
    assert not _value_named_by_user(
        "cvat", _BACKEND_USER_TOKENS, ["cvat or label studio?", "ok"])

    state = WorkflowState(workflow_name="auto_labeling", dataset_name="ds", dataset_confirmed=True,
                          auto_labeling=AutoLabelingState(labeling_backend="both"))
    result, routings = asyncio.run(_pipeline(state, ["try again"])._handle_set_labeling_backend(
        {"backend": "cvat"}, _client()))
    assert result == Sentinels.BACKEND_NOT_NAMED and isinstance(routings[0], HardStop)


def test_path_ordinal_does_not_match_dataset_name():
    state = WorkflowState(workflow_name="auto_labeling", dataset_name="ds", dataset_confirmed=True,
                          auto_labeling=AutoLabelingState(labeling_backend="cvat"))
    result, _ = asyncio.run(_pipeline(state, ["use custom_dataset1"])._handle_set_labeling_path(
        {"path": "manual"}))
    assert result == Sentinels.PATH_NOT_NAMED


def test_confirm_without_pending_summary_is_blocked():
    state = WorkflowState(workflow_name="auto_labeling", dataset_name="ds", dataset_confirmed=True,
                          auto_labeling=AutoLabelingState())
    p = _pipeline(state, ["yes"])
    for handler in (p._handle_confirm_export, p._handle_confirm_auto_labeling_run):
        result, routings = asyncio.run(handler())
        assert result == Sentinels.CONFIRM_NOT_PENDING and isinstance(routings[0], Injection)
    assert not state.auto_labeling.export_confirmed and not state.auto_labeling.run_confirmed

    state.auto_labeling.run_awaiting_confirmation = True
    asyncio.run(p._handle_confirm_auto_labeling_run())
    assert state.auto_labeling.run_confirmed
