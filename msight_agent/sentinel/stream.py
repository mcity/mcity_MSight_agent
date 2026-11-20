"""Publishes incident stages to the live feed (msight/runtime/incidents.py), which the
chat page shows as they happen. The feed lives in Redis, so during a Redis outage
events are held (and logged to incidents.jsonl as usual) and published in order once
it's back -- the chat server shows its own "Redis unreachable" notice meanwhile. Only
the newest HELD_EVENTS are kept through a very long outage. friendly_step words each
of Claude's steps for the feed."""
from collections import deque
from pathlib import Path
from typing import Callable

import redis

from msight_agent.msight.runtime.incidents import STREAM_KEY, STREAM_MAXLEN, encode


HELD_EVENTS = 1000


class IncidentPublisher:
    def __init__(self, settings: dict, say: Callable[[str], None]):
        self._client = redis.Redis(**settings, socket_timeout=2, socket_connect_timeout=2)
        self._say = say
        self._held: deque[dict] = deque(maxlen=HELD_EVENTS)
        self._dropped = 0

    def __call__(self, incident_id: str, stage: str, **fields) -> None:
        if len(self._held) == HELD_EVENTS:
            self._dropped += 1          # the deque drops the oldest held event
        self._held.append(encode(incident_id, stage, fields))
        failing_before = len(self._held) > 1
        try:
            while self._held:
                self._client.xadd(STREAM_KEY, self._held[0], maxlen=STREAM_MAXLEN, approximate=True)
                self._held.popleft()
            if failing_before:
                lost = f"; the oldest {self._dropped} were dropped (still in incidents.jsonl)" if self._dropped else ""
                self._say(f"[feed] Redis is back; held incident events were published{lost}")
                self._dropped = 0
        except redis.RedisError as e:
            if not failing_before:  # once per outage, not once per event
                self._say(f"[feed] can't publish to the chat feed ({e}); holding events until Redis is back")


def friendly_step(tool_name: str, tool_input: dict) -> str:
    """One investigation step in plain words, for the chat."""
    name = tool_name.split("__")[-1]
    if name == "diagnose_msight_pipeline":
        return "Measuring live data flow on every topic"
    if name == "get_msight_status":
        return "Checking which nodes are running"
    if name == "get_msight_logs":
        return f"Reading the logs of {tool_input.get('service') or 'the pipeline'}"
    if name == "restart_msight_node":
        return f"Restarting {tool_input.get('name') or 'the node'} with its normal settings"
    if name == "start_msight_recording":
        return "Restarting the recording"
    if name == "inspect_host":
        command = " ".join(str(tool_input.get("command", "")).split())
        return f"Running `{command[:70]}`"
    if name == "Read":
        return f"Reading {Path(str(tool_input.get('file_path', ''))).name}"
    if name == "Grep":
        return f"Searching the code for “{str(tool_input.get('pattern', ''))[:50]}”"
    if name == "Glob":
        return "Looking through the source files"
    return name.replace("_", " ")
