"""Graph diagnosis: live topic rates + the publisher -> topic -> subscriber graph.

Pure functions over node_health() rows (see MSightControlPlane), plus the
Redis rate probe they're computed from.
"""
import difflib
import time
from typing import Optional

import redis

from mcp_layer.mcptools.msight.utils import ago, redis_kwargs

BROKEN = {"DEAD", "STALLED", "UNREGISTERED"}

# User-facing wording: no tool names (base_prompt.txt forbids showing them).
_FIXES = {
    "DEAD": "Restart that node with the same settings (or restart the whole pipeline "
            "if it's one of the standard pipeline's nodes).",
    "STALLED": "It's receiving input but producing nothing -- check its logs for an error, "
               "then restart it.",
    "UNREGISTERED": "It's running but never finished registering -- check its logs, and "
                    "restart it if it's stuck starting up.",
    "MISSING_PUBLISHER": "Add back the node that should publish this topic, or restart the "
                         "pipeline if it was one of the standard pipeline's nodes.",
    "TOPIC_MISMATCH": "Remove that node and add it back with the corrected input topic.",
}


def _publishers(rows: list[dict]) -> dict[str, list[dict]]:
    by_topic: dict[str, list[dict]] = {}
    for r in rows:
        for t in r.get("publish_topics", []):
            by_topic.setdefault(t, []).append(r)
    return by_topic


def structural_findings(rows: list[dict]) -> list[dict]:
    """Cheap graph checks (registry + liveness, no rate probe).

    DEAD               tracked node isn't running
    MISSING_PUBLISHER  a running node subscribes to a topic nobody publishes
    TOPIC_MISMATCH     same, but a published topic is a near-match
    """
    pubs = _publishers(rows)
    findings = [
        {"kind": "DEAD", "node": r["name"],
         "evidence": f"'{r['name']}' is registered/tracked but its process/container is not running."}
        for r in rows if not r["alive"]
    ]
    for r in rows:
        if not r["alive"]:
            continue
        for t in r.get("subscribe_topics", []):
            if pubs.get(t):
                continue  # has a publisher (a dead one is its own finding)
            close = difflib.get_close_matches(t, list(pubs), n=1, cutoff=0.75)
            if close:
                findings.append({
                    "kind": "TOPIC_MISMATCH", "node": r["name"], "topic": t,
                    "evidence": (f"'{r['name']}' subscribes to '{t}', which nothing publishes; "
                                 f"the closest published topic is '{close[0]}'."),
                })
            else:
                findings.append({
                    "kind": "MISSING_PUBLISHER", "node": r["name"], "topic": t,
                    "evidence": (f"'{r['name']}' subscribes to '{t}', but no registered node publishes "
                                 "it -- its upstream node was removed, crashed and deregistered, "
                                 "or was never started."),
                })
    return findings


def _last_removed_publisher(topic: str, changes: list[dict]) -> Optional[dict]:
    return next((c for c in reversed(changes)
                 if c["change"] == "removed" and topic in c.get("publish_topics", [])), None)


def _broken_node_evidence(r: dict) -> str:
    if r["health"] == "DEAD":
        return f"'{r['name']}' is not running."
    if r["health"] == "UNREGISTERED":
        return f"'{r['name']}' is running but missing from the node registry."
    rate = r["in_rate_hz"] if r["in_rate_hz"] is not None else "n/a (source)"
    return f"'{r['name']}' is running and receiving {rate} msg/s, but published nothing during the sample."


def _summary(rows: list[dict], root_causes: list[dict], affected: list[dict]) -> str:
    if not root_causes and not affected:
        return f"All {len(rows)} nodes are running and data is flowing."
    if not root_causes:
        return "Some nodes are starved of input but no root cause was identified -- see nodes."
    summary = f"Most likely root cause: {root_causes[0]['evidence']}"
    if affected:
        summary += (f" {len(affected)} downstream node(s) are affected as a consequence: "
                    f"{', '.join(a['node'] for a in affected)}.")
    return summary


def diagnose_from_health(rows: list[dict], changes: Optional[list[dict]] = None) -> dict:
    """Rank root causes vs downstream effects from node_health() rows.

    A broken node is a root cause only if nothing upstream is broken, since
    msight_core nodes exit after ~15s without input and failures cascade.
    `changes` (oldest first) lets an orphaned topic name the node whose
    removal orphaned it.
    """
    if not rows:
        return {"verdict": "empty", "summary": "No MSight nodes are registered or tracked.",
                "root_causes": [], "affected": [], "nodes": []}

    changes = changes or []
    by_name = {r["name"]: r for r in rows}
    pubs = _publishers(rows)

    def upstream_broken(r: dict, seen: set) -> Optional[str]:
        for t in r.get("subscribe_topics", []):
            for p in pubs.get(t, []):
                if p["name"] in seen:
                    continue
                seen.add(p["name"])
                deeper = upstream_broken(p, seen)
                if deeper:
                    return deeper
                if p["health"] in BROKEN:
                    return p["name"]
        return None

    def depth(r: dict, seen: set) -> int:
        ups = [p for t in r.get("subscribe_topics", []) for p in pubs.get(t, []) if p["name"] not in seen]
        return 0 if not ups else 1 + max(depth(p, seen | {p["name"]}) for p in ups)

    root_causes, affected = [], []
    for r in rows:
        if r["health"] not in BROKEN:
            continue
        culprit = upstream_broken(r, {r["name"]})
        if culprit:
            affected.append({"node": r["name"], "health": r["health"], "caused_by": culprit})
        else:
            root_causes.append({"kind": r["health"], "node": r["name"],
                                "evidence": _broken_node_evidence(r), "_depth": depth(r, {r["name"]})})

    for f in structural_findings(rows):
        if f["kind"] == "DEAD":
            continue  # already covered by health above
        cause = {**f, "_depth": depth(by_name[f["node"]], {f["node"]})}
        removed = _last_removed_publisher(f["topic"], changes) if f["kind"] == "MISSING_PUBLISHER" else None
        if removed:
            cause["evidence"] = (f"'{f['node']}' subscribes to '{f['topic']}', which was published by "
                                 f"'{removed['node']}' -- removed {ago(time.time() - removed['at'])}.")
            cause["removed_publisher"] = removed["node"]
        root_causes.append(cause)

    for r in rows:
        if r["health"] != "STARVED":
            continue
        culprit = upstream_broken(r, {r["name"]})
        if not culprit:
            if any(c["node"] == r["name"] for c in root_causes):
                continue  # its own missing/mismatched publisher is already the root cause
            culprit = "unknown -- see nodes"
        affected.append({"node": r["name"], "health": "STARVED", "caused_by": culprit})

    root_causes.sort(key=lambda c: c.pop("_depth"))
    for c in root_causes:
        c["suggested_fix"] = (
            f"Add '{c['removed_publisher']}' back (or restart the pipeline if it was one of the "
            "standard pipeline's nodes)." if c.get("removed_publisher") else _FIXES[c["kind"]]
        )

    return {
        "verdict": "healthy" if not root_causes and not affected else "degraded",
        "summary": _summary(rows, root_causes, affected),
        "root_causes": root_causes,
        "affected": affected,
        "nodes": rows,
    }


def sample_topic_rates(topics: list[str], max_window: float) -> dict[str, dict]:
    """Per-topic message rate, measured by subscribing to each topic's Redis channel.

    Stops once every topic has delivered, else at max_window, which must exceed
    the slowest healthy interval (RF-DETR on CPU: ~4.5s).
    """
    if not topics:
        return {}
    client = redis.Redis(**redis_kwargs())
    pubsub = client.pubsub(ignore_subscribe_messages=True)
    counts = dict.fromkeys(topics, 0)
    start = time.monotonic()
    try:
        pubsub.subscribe(*topics)
        deadline = start + max_window
        while (remaining := deadline - time.monotonic()) > 0:
            msg = pubsub.get_message(timeout=remaining)
            if not msg or msg["type"] != "message":
                continue
            channel = msg["channel"].decode(errors="replace")
            if channel in counts:
                counts[channel] += 1
                if all(counts.values()):
                    # Listen >=1s so one early message doesn't inflate the rate.
                    deadline = min(deadline, max(start + 1.0, time.monotonic()))
    finally:
        pubsub.close()
        client.close()
    elapsed = max(time.monotonic() - start, 1e-6)
    return {
        t: {"messages": c, "rate_hz": round(c / elapsed, 2), "window_s": round(elapsed, 1)}
        for t, c in counts.items()
    }
