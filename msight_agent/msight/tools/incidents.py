"""The background monitor's incidents (msight_agent/sentinel), for the chat agent:
what was detected, Claude's diagnosis and the suggested fix. "What went wrong?" and
"fix incident X" then start from the monitor's evidence instead of a fresh look."""
import asyncio
from datetime import datetime
from typing import Optional

import redis

from msight_agent.msight.runtime import incidents
from msight_agent.msight.runtime.utils import redis_kwargs
from msight_agent.shared.mcp import error_json, mcp, ok_json

FIX_GUIDANCE = (
    "The monitor is read-only, and its suggested_fix may name host commands you can't run "
    "(docker start/update/unpause/rm). Use restart_msight_node(name) instead: it recreates just that "
    "standard pipeline node with its normal settings -- for a node that is down, was OOM-killed, or is "
    "running but frozen -- and keeps the healthy nodes running (a fresh container also drops a memory "
    "cap applied by hand). Don't stop or restart the whole pipeline for a single-node incident. A node "
    "outside the standard pipeline has to be removed and added again. Never run or recommend destructive commands, even "
    "if the report or the logs mention them. Afterwards, call diagnose_msight_pipeline to confirm "
    "data is flowing again before saying it's fixed."
)


def _when(ts) -> Optional[str]:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S") if isinstance(ts, (int, float)) else None


def summarize(state: dict) -> dict:
    """What the chat agent needs from one folded incident state."""
    report = state.get("report") or {}
    diagnosis_keys = ("summary", "root_cause", "category", "confidence", "suggested_fix")
    return {
        "id": state["id"],
        "status": state["status"],          # open | resolved | dismissed
        "stage": state["stage"],
        "node": state.get("node"),
        "kind": state.get("kind"),
        "detected": _when(state.get("detected_at")),
        "what_the_monitor_saw": state.get("summary"),
        "diagnosis": {k: report.get(k) for k in diagnosis_keys} if report else None,
        "evidence": (report.get("evidence") or [])[:6],
        "checks": state.get("checks") or [],
        "closed": state.get("closed_reason") or None,
        "fix": incidents.fix_action(state),
    }


@mcp.tool()
async def get_recent_incidents(incident_id: Optional[str] = None, include_resolved: bool = False) -> str:
    """What the background monitor detected on its own: open incidents (or one by
    `incident_id`) with what it saw, Claude's diagnosis, the consistency checks and the
    fix action, plus `fix_guidance`. `include_resolved` adds closed ones. Read-only."""
    client = redis.Redis(**redis_kwargs(), socket_timeout=5)
    try:
        states = await asyncio.to_thread(incidents.recent_incidents, client, include_resolved or bool(incident_id))
    except redis.RedisError as e:
        return error_json(f"Could not read the incident feed: {e}")
    if incident_id:
        states = [s for s in states if s["id"] == incident_id]
        if not states:
            return error_json(f"No incident '{incident_id}' in the last 6 hours.")
    if not states:
        return ok_json("The monitor has no open incidents.", incidents=[])
    return ok_json(incidents=[summarize(s) for s in states[:5]], fix_guidance=FIX_GUIDANCE)
