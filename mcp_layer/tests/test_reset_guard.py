"""
The confirmation guard on reset_workflow_state.

reset_workflow_state discards the workflow, the dataset and every flag in one
call, and it lives in ALWAYS_TOOLS, so it stays callable in every phase --
including the locked phases where switch_workflow is refused without
confirm_restart. It had no guard at all: a single "reset workflow" wiped a live
session with no question asked.

The guard is a state flag, not a tool argument, so the model cannot assert the
user's consent by filling in a parameter. It is deliberately short-lived: the
turn that asks is the only turn that can answer.

WorkflowState.save is patched throughout: these tests must not write config.py.
"""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from chat_pipeline import ChatPipeline
from pipeline_common import HardStop, Sentinels
from validate_workflow_state import (
    ALWAYS_TOOLS, AutoLabelingPhase, AutoLabelingState, WorkflowState,
)


def _live_session() -> WorkflowState:
    """A session with real work in it: workflow, dataset, model, backend."""
    return WorkflowState(
        workflow_name="auto_labeling",
        dataset_name="custom_dataset6",
        dataset_confirmed=True,
        auto_labeling=AutoLabelingState(
            labeling_backend="cvat",
            labeling_path="auto",
            model_name="yolo12x",
            model_configured=True,
        ),
    )


def _reset(state: WorkflowState) -> tuple[str, list, MagicMock, ChatPipeline]:
    pipeline = ChatPipeline(mcp_client=MagicMock(), llm=MagicMock())
    pipeline.state = state
    client = MagicMock()
    client.call_tool = AsyncMock(return_value="Workflow, dataset, and session state have been reset.")
    with patch.object(WorkflowState, "save"):
        result, routings = asyncio.run(pipeline._handle_reset_workflow_state(client))
    return result, routings, client, pipeline


# --- the guard itself -------------------------------------------------------


def test_first_call_asks_and_clears_nothing():
    state = _live_session()
    result, routings, client, pipeline = _reset(state)

    client.call_tool.assert_not_called()
    assert Sentinels.RESET_NEEDS_CONFIRMATION in result
    assert isinstance(routings[0], HardStop)
    # Every value the user set up survives the question.
    assert pipeline.state.workflow_name == "auto_labeling"
    assert pipeline.state.dataset_confirmed is True
    assert pipeline.state.auto_labeling.model_name == "yolo12x"
    assert pipeline.state.reset_awaiting_confirmation is True
    assert pipeline.state.reset_confirmation_requested_at > 0


def test_second_call_carries_it_out():
    state = _live_session()
    state.reset_awaiting_confirmation = True
    state.reset_confirmation_requested_at = time.time()

    result, routings, client, pipeline = _reset(state)

    client.call_tool.assert_called_once_with("reset_workflow_state", {})
    assert isinstance(routings[0], HardStop)
    assert pipeline.state.workflow_name == ""
    assert pipeline.state.dataset_confirmed is False
    assert pipeline.state.auto_labeling is None
    assert pipeline.state.reset_awaiting_confirmation is False


def test_empty_session_resets_without_asking():
    """Nothing to lose, so the question would be a pointless round trip."""
    result, routings, client, pipeline = _reset(WorkflowState())

    client.call_tool.assert_called_once_with("reset_workflow_state", {})
    assert Sentinels.RESET_NEEDS_CONFIRMATION not in result


def test_guard_applies_in_a_locked_phase():
    """switch_workflow is refused here without confirm_restart. reset_workflow_state
    does strictly more damage and must not be the way around that lock."""
    state = _live_session()
    state.auto_labeling.phase = AutoLabelingPhase.TRAINING

    result, routings, client, pipeline = _reset(state)

    assert "reset_workflow_state" in ALWAYS_TOOLS      # still offered while locked
    client.call_tool.assert_not_called()
    assert Sentinels.RESET_NEEDS_CONFIRMATION in result
    assert pipeline.state.auto_labeling.phase == AutoLabelingPhase.TRAINING


# --- the reply the user sees ------------------------------------------------


def test_question_names_no_tool():
    """The HardStop becomes an assistant turn and returns to the model for the
    next four requests. A tool name in it reads as a standing order -- the
    failure test_locked_reply.py was written for."""
    _, routings, _, _ = _reset(_live_session())
    reply = routings[0].reply

    for tool in ("reset_workflow_state", "switch_workflow", "select_workflow"):
        assert tool not in reply
    assert "()" not in reply


def test_question_lists_what_would_be_lost():
    _, routings, _, _ = _reset(_live_session())
    reply = routings[0].reply

    assert "custom_dataset6" in reply
    assert "yolo12x" in reply
    assert "CVAT" in reply
    assert "Nothing has been cleared yet" in reply


# --- the pending flag does not outlive its turn -----------------------------


def _run_with_tool(state: WorkflowState, tool_name: str) -> WorkflowState:
    """Drive ChatPipeline.run() for one non-reset tool call."""
    pipeline = ChatPipeline(mcp_client=MagicMock(), llm=MagicMock())
    call = MagicMock()
    call.id = "call_1"
    call.function.name = tool_name
    call.function.arguments = "{}"

    with patch.object(WorkflowState, "load", return_value=state), \
         patch.object(WorkflowState, "save"), \
         patch.object(ChatPipeline, "_dispatch", new=AsyncMock(return_value=("ok", []))):
        asyncio.run(pipeline.run([call], []))
    return pipeline.state


def test_pending_reset_is_dropped_when_the_turn_does_something_else():
    state = _live_session()
    state.reset_awaiting_confirmation = True
    state.reset_confirmation_requested_at = time.time()

    ending = _run_with_tool(state, "send_reply")

    assert ending.reset_awaiting_confirmation is False
    assert ending.reset_confirmation_requested_at == 0.0


def test_ttl_clears_a_reset_nobody_answered():
    """load() drops a confirmation older than 5 minutes, so a much later "yes"
    never lands on an open gate."""
    stale = _live_session().model_dump()
    stale["reset_awaiting_confirmation"] = True
    stale["reset_confirmation_requested_at"] = time.time() - 400

    with patch("validate_workflow_state.importlib.reload"), \
         patch("validate_workflow_state._cc") as cc:
        cc.WORKFLOW_STATE = stale
        loaded = WorkflowState.load()

    assert loaded.reset_awaiting_confirmation is False
    assert loaded.workflow_name == "auto_labeling"   # the session itself survives


def test_migrate_keeps_the_reset_fields():
    """_migrate drops every key outside its `known` set. A field missing from
    that set is silently reset on every load -- the flag would never survive
    the round trip to config.py."""
    raw = WorkflowState(
        workflow_name="auto_labeling",
        reset_awaiting_confirmation=True,
        reset_confirmation_requested_at=1234.0,
    ).model_dump()

    migrated = WorkflowState._migrate(raw)

    assert migrated["reset_awaiting_confirmation"] is True
    assert migrated["reset_confirmation_requested_at"] == 1234.0
