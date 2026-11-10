"""The live incident feed shared by the sentinel (writer), the chat server (pushes it
to the browser) and the chat agent (reads it).

The sentinel appends each stage of an incident to a Redis Stream as it happens:

  detected       seconds after the fault, provisional ("X just died, checking...")
  classified     the watcher's final kind, summary and timeline
  investigating  Claude has started
  progress       one per investigation step
  diagnosed      Claude's report and the code-side checks
  resolved       the pipeline is healthy again
  dismissed      not a fault after all (removed on purpose, recovered on its own)

fold() merges a stream's events into one current state per incident. Takes a redis
client instead of creating one, so it works with either venv's redis-py.
"""
import json
import time
from typing import Iterable, Optional

STREAM_KEY = "MSIGHT:INCIDENTS"
STREAM_MAXLEN = 2000
CLOSED = ("resolved", "dismissed")
_MAX_STEPS = 12


def encode(incident_id: str, stage: str, fields: dict) -> dict:
    return {"data": json.dumps({"incident": incident_id, "stage": stage, "at": time.time(), **fields},
                               default=str)}


def decode(entry: dict) -> Optional[dict]:
    raw = entry.get("data") or entry.get(b"data")
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def apply(state: Optional[dict], event: dict) -> dict:
    """One incident's state after one more event."""
    state = dict(state or {"id": event["incident"], "status": "open", "stage": "detected",
                           "detected_at": event["at"], "steps": []})
    stage = event["stage"]
    state.update({k: v for k, v in event.items() if k not in ("incident", "stage", "at", "text")})
    state["updated_at"] = event["at"]
    if stage == "progress":
        state["steps"] = (state["steps"] + [event.get("text", "")])[-_MAX_STEPS:]
        return state
    if stage in CLOSED:
        state["status"] = stage
        state[f"{stage}_at"] = event["at"]
        state["closed_reason"] = event.get("text", "")
    # A diagnosis can land after the pipeline already recovered: keep it closed.
    state["stage"] = stage if state["status"] == "open" or stage in CLOSED else state["stage"]
    return state


def fold(events: Iterable[dict]) -> dict[str, dict]:
    states: dict[str, dict] = {}
    for event in events:
        if event and event.get("incident"):
            states[event["incident"]] = apply(states.get(event["incident"]), event)
    return states


_DOWN = {"CONTAINER_OOM", "CONTAINER_KILLED", "CONTAINER_CRASHED", "CONTAINER_EXITED", "PROCESS_DIED"}
_STUCK = {"NODE_UNRESPONSIVE", "PIPELINE_DEGRADED"}


def fix_action(state: dict) -> Optional[dict]:
    """{"label", "message"} for the chat's Fix-it button, or None when the chat agent
    can't fix it (a service outage) or there's nothing to fix yet.

    The message is what the user sends by clicking, so it names the action itself:
    the chat's write guards only let the agent remove or stop something the user's
    own message asked for."""
    if state.get("status") != "open" or not state.get("report"):
        return None
    node, kind, incident_id = state.get("node", ""), state.get("kind"), state["id"]
    if kind in _DOWN or (kind in _STUCK and "+" not in node):
        return {"label": f"Fix it: restart {node}",
                # No "restart"/"stop": the guards would read those as consent to stop the whole pipeline.
                "message": f"Fix incident {incident_id}: recreate {node} with its normal settings and keep "
                           "the rest of the pipeline running."}
    if kind in _STUCK:
        return {"label": "Fix it: restart the pipeline",
                "message": f"Fix incident {incident_id}: restart the pipeline -- stop it and start it again so "
                           "every node comes back with its normal settings."}
    return None


def read_recent(client, count: int = 500) -> list[tuple[str, dict]]:
    """[(stream id, event)] oldest first."""
    entries = client.xrevrange(STREAM_KEY, count=count)
    out = []
    for stream_id, fields in reversed(entries):
        event = decode(fields)
        if event:
            out.append((stream_id.decode() if isinstance(stream_id, bytes) else stream_id, event))
    return out


def recent_incidents(client, include_closed: bool = False, max_age_s: float = 6 * 3600) -> list[dict]:
    """Newest first; open ones only unless include_closed."""
    now = time.time()
    states = fold(event for _, event in read_recent(client)).values()
    return sorted((s for s in states if (include_closed or s["status"] == "open")
                   and now - s["updated_at"] < max_age_s),
                  key=lambda s: s["detected_at"], reverse=True)
