"""Publishes incident stages to the live feed (msight/runtime/incidents.py), which the
chat page shows as they happen. A Redis outage never stops monitoring: events are
then only in the console and incidents.jsonl."""
import os
from pathlib import Path
from typing import Callable

import redis

from msight_agent.msight.runtime.incidents import STREAM_KEY, STREAM_MAXLEN, encode


def redis_settings() -> dict:
    """The MSight broker settings (same variables as msight/runtime/utils.redis_kwargs).
    Read before the sentinel scrubs .env values from its environment."""
    return {
        "host": os.environ.get("MSIGHT_REDIS_MESSAGE_BROKER_HOST", "localhost"),
        "port": int(os.environ.get("MSIGHT_REDIS_MESSAGE_BROKER_PORT", 6379)),
        "db": int(os.environ.get("MSIGHT_REDIS_MESSAGE_BROKER_DB", 0)),
    }


class IncidentPublisher:
    def __init__(self, settings: dict, say: Callable[[str], None]):
        self._client = redis.Redis(**settings, socket_timeout=2, socket_connect_timeout=2)
        self._say = say
        self._failing = False

    def __call__(self, incident_id: str, stage: str, **fields) -> None:
        try:
            self._client.xadd(STREAM_KEY, encode(incident_id, stage, fields), maxlen=STREAM_MAXLEN, approximate=True)
            if self._failing:
                self._say("[feed] Redis is back; the chat feed is live again")
            self._failing = False
        except redis.RedisError as e:
            if not self._failing:  # once per outage, not once per event
                self._say(f"[feed] can't publish to the chat feed ({e}); incidents are still logged")
            self._failing = True


def friendly_step(tool_name: str, tool_input: dict) -> str:
    """One investigation step in plain words, for the chat."""
    name = tool_name.split("__")[-1]
    if name == "diagnose_msight_pipeline":
        return "Measuring live data flow on every topic"
    if name == "get_msight_status":
        return "Checking which nodes are running"
    if name == "get_msight_logs":
        return f"Reading the logs of {tool_input.get('service') or 'the pipeline'}"
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
