"""A failed auto-labeling run, or one whose export delivered 0 images, must leave the
workflow in Zone A (editable, retryable) instead of locking it."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from msight_agent.agent.dispatch import ChatPipeline
from msight_agent.agent.state import AutoLabelingPhase, AutoLabelingState, WorkflowState
from msight_agent.auto_labeling.handlers import (
    _parse_cvat_task_ids, export_empty, parse_run_failure, run_failed,
)

FAILED = (
    "RUN_FAILED: Auto-labeling failed with exit code 1.\n"
    "Last error line: CUDA out of memory\n"
    "Full logs saved to `/logs/last_auto_labeling_log.txt`"
)
SUCCESS = "Auto-labeling workflow completed.\n\n**Result Summary:**\n```\nok\n```"


@pytest.fixture
def pipeline():
    p = ChatPipeline(mcp_client=MagicMock(), llm=MagicMock())
    p.state = WorkflowState(
        workflow_name="auto_labeling", dataset_name="ds", dataset_confirmed=True,
        auto_labeling=AutoLabelingState(
            labeling_path="auto", labeling_backend="cvat", model_configured=True,
            hyperparams_confirmed=True, run_confirmed=True,
        ),
    )
    p.llm.summarize_classification_report = AsyncMock(return_value="")
    with patch.object(WorkflowState, "save"):
        yield p


def _client(export_reply: str):
    client = MagicMock()
    client.call_tool = AsyncMock(return_value=export_reply)
    return client


def test_failure_helpers():
    assert run_failed(FAILED)
    assert run_failed("Auto-labeling failed with exit code 2.\nError details: ...")
    assert not run_failed(SUCCESS + "\nsome log: failed with exit code 0")
    assert parse_run_failure(FAILED) == ("CUDA out of memory", "/logs/last_auto_labeling_log.txt")
    streamed = "RUN_FAILED: Auto-labeling failed with exit code 1.\nError details:\n```\nfoo\nValueError: bad\n```\n"
    assert parse_run_failure(streamed) == ("ValueError: bad", "")
    assert export_empty("EXPORT_NO_IMAGES: Dataset 'ds' contains 0 images")
    assert _parse_cvat_task_ids("Exported.\nTask IDs: 4, 5, 6\nImages: 30\n") == [4, 5, 6]


def test_failed_run_stays_editable(pipeline):
    reply = asyncio.run(pipeline._finalize_auto_labeling(FAILED, _client("")))
    al = pipeline.state.auto_labeling
    assert al.phase == AutoLabelingPhase.PENDING and not al.auto_labeling_complete
    assert not al.run_confirmed                      # retry shows a fresh summary
    assert "CUDA out of memory" in reply
    lr = pipeline.state.failed_run()
    assert lr and lr.attempts == 1

    asyncio.run(pipeline._finalize_auto_labeling(FAILED, _client("")))
    assert pipeline.state.failed_run().attempts == 2


def test_empty_export_marks_run_failed(pipeline):
    reply = asyncio.run(pipeline._finalize_auto_labeling(
        SUCCESS, _client("EXPORT_NO_IMAGES: Dataset 'ds' contains 0 images")))
    al = pipeline.state.auto_labeling
    assert al.phase == AutoLabelingPhase.PENDING and not al.auto_labeling_complete
    assert "0 images" in reply
    assert pipeline.state.failed_run() is not None


def test_successful_run_locks_and_records_task_ids(pipeline):
    asyncio.run(pipeline._finalize_auto_labeling(
        SUCCESS, _client("Exported.\nTask IDs: 7, 8\nImages: 30\n")))
    al = pipeline.state.auto_labeling
    assert al.phase == AutoLabelingPhase.TRAINING and al.auto_labeling_complete
    assert al.cvat_task_ids == [7, 8]
    assert pipeline.state.failed_run() is None
    assert pipeline.state.last_run.status == "success"


def test_failed_run_hint_survives_migrate():
    raw = {"workflow_name": "auto_labeling",
           "last_run": {"workflow": "auto_labeling", "status": "failed", "error": "x",
                        "attempts": 1, "stale_key": True}}
    state = WorkflowState.model_validate(WorkflowState._migrate(raw))
    assert state.failed_run().error == "x"


def test_export_error_keeps_predictions_and_offers_export_only(pipeline):
    limit = ("CVAT_TASK_LIMIT_REACHED: This CVAT account allows 3 tasks and already has 3, "
             "so there is no room for 'ds'. Delete some existing tasks.")
    reply = asyncio.run(pipeline._finalize_auto_labeling(SUCCESS, _client(limit)))
    al = pipeline.state.auto_labeling
    assert al.auto_labeling_complete and al.phase == AutoLabelingPhase.PENDING
    assert pipeline.state.failed_run() is None          # the run itself succeeded
    assert "CVAT_TASK_LIMIT_REACHED" not in reply and "already has 3" in reply
    assert "0 images" not in reply and "export again" in reply


def test_retrying_the_export_sends_predictions_without_rerunning(pipeline, monkeypatch):
    pipeline.state.auto_labeling.auto_labeling_complete = True
    monkeypatch.setenv("CVAT_ACCESS_TOKEN", "tok")
    client = _client("Exported.\nTask IDs: 9\nImages: 30\n")
    asyncio.run(pipeline._handle_export_generic("cvat", {"dataset_name": "ds"}, client))
    args = client.call_tool.call_args[0]
    assert args[0] == "export_to_cvat" and args[1]["with_predictions"] is True
    al = pipeline.state.auto_labeling
    assert al.cvat_task_ids == [9] and al.phase == AutoLabelingPhase.TRAINING
