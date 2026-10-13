"""Agent turn guards: diagnose-then-act consent gate, measure-first gate and LLM retry."""
import asyncio

import pytest

from mcp_layer.chat.guards import (
    CHANGE_REQUEST_RE, HEALTH_QUESTION_RE, chat_with_retry, measure_first, without_writes,
)


@pytest.mark.parametrize("message", [
    "the viewer is frozen",
    "why am I not seeing any detections?",
    "the viewer page stopped loading",
    "is the pipeline actually working right now?",
])
def test_symptom_reports_are_not_change_requests(message):
    # "stopped" is a symptom, not the verb "stop".
    assert not CHANGE_REQUEST_RE.search(message)


@pytest.mark.parametrize("message", [
    "the viewer is frozen, fix it",
    "restart the detector",
    "yes",
    "go ahead",
    "remove viewer_2 and add it back",
])
def test_change_requests_are_recognised(message):
    assert CHANGE_REQUEST_RE.search(message)


def test_without_writes_keeps_read_tools():
    tools = [{"function": {"name": n}} for n in
             ("diagnose_msight_pipeline", "add_msight_node", "get_msight_logs", "stop_msight_pipeline")]
    assert [t["function"]["name"] for t in without_writes(tools)] == \
        ["diagnose_msight_pipeline", "get_msight_logs"]


class APIConnectionError(Exception):
    pass


class _FlakyClient:
    def __init__(self, failures, exc=APIConnectionError):
        self.calls, self.failures, self.exc = 0, failures, exc

    async def chat(self, messages, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc("boom")
        return "ok"


def test_retry_recovers_from_one_dropped_connection():
    client = _FlakyClient(failures=1)
    assert asyncio.run(chat_with_retry(client, [])) == "ok"
    assert client.calls == 2


def test_retry_gives_up_after_second_failure():
    client = _FlakyClient(failures=2)
    with pytest.raises(APIConnectionError):
        asyncio.run(chat_with_retry(client, []))
    assert client.calls == 2


def test_non_connection_errors_are_not_retried():
    client = _FlakyClient(failures=1, exc=ValueError)
    with pytest.raises(ValueError):
        asyncio.run(chat_with_retry(client, []))
    assert client.calls == 1


@pytest.mark.parametrize("message, expected", [
    ("is the pipeline actually working right now?", True),
    ("is it running on cpu or gpu ?", True),
    ("is everything healthy?", True),
    ("check status", False),
    ("what nodes are running?", False),
    ("add a detection viewer on port 9011", False),
])
def test_health_questions_are_recognised(message, expected):
    assert bool(HEALTH_QUESTION_RE.search(message)) is expected


def test_measure_first_offers_only_diagnose_when_available():
    with_diag = [{"function": {"name": n}} for n in
                 ("get_msight_status", "get_msight_logs", "diagnose_msight_pipeline")]
    assert [t["function"]["name"] for t in measure_first(with_diag)] == ["diagnose_msight_pipeline"]
    status_only = [{"function": {"name": "get_msight_status"}}]
    assert measure_first(status_only) == status_only
