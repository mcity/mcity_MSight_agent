"""Soak test of the sentinel: leave it watching a healthy pipeline for hours, with no faults
injected, and record what a long run does to it.

Every sample (default 5 min) it notes
  false alarms   incidents opened while nothing was wrong (each one is listed)
  sentinel       memory (RSS), threads and open files, so a leak shows as a steady climb
  redis          connected clients, to catch connections that are never closed
  pipeline       the /diagnose verdict, so a real failure during the run isn't miscounted

Needs mcp_server, msight_api, chat_server, a running pipeline and the sentinel (any autonomy).

    PYTHONPATH=. .venv/bin/python3 -m msight_agent.evals.sentinel_soak --hours 8
"""
import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Optional

import redis

from msight_agent.evals.sentinel_e2e import Feed
from msight_agent.shared.paths import LOG_DIR

API = "http://localhost:8003"
OUT_DIR = LOG_DIR / "sentinel_soak"


def sentinel_pid() -> Optional[int]:
    """The sentinel process (python -m msight_agent.sentinel, not one of its submodules)."""
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if (b"python" in Path(argv[0].decode(errors="replace")).name.encode() and b"-m" in argv
                and argv[argv.index(b"-m") + 1:argv.index(b"-m") + 2] == [b"msight_agent.sentinel"]):
            return int(proc.name)
    return None


def process_stats(pid: int) -> dict:
    status = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines() if ":" in line)
    return {"rss_mb": round(int(status["VmRSS"].split()[0]) / 1024, 1), "threads": int(status["Threads"]),
            "open_files": len(os.listdir(f"/proc/{pid}/fd"))}


def verdict() -> str:
    try:
        with urllib.request.urlopen(API + "/diagnose", timeout=60) as resp:
            return json.loads(resp.read()).get("verdict", "?")
    except Exception as e:  # noqa: BLE001
        return f"unreachable ({type(e).__name__})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--every", type=float, default=300, help="seconds between samples")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"soak-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    feed, client, t0 = Feed(), redis.Redis(socket_timeout=10), time.time()
    first_pid, alarms = sentinel_pid(), {}
    print(f"Soak test for {args.hours}h, a sample every {args.every:.0f}s -> {out}", flush=True)
    while time.time() - t0 < args.hours * 3600:
        deadline = time.time() + args.every
        while time.time() < deadline:
            feed.poll(1000)
        pid = sentinel_pid()
        for state in feed.since(t0):
            alarms.setdefault(state["id"], state)
        sample = {"at": time.strftime("%H:%M:%S"), "elapsed_h": round((time.time() - t0) / 3600, 2),
                  "pipeline": verdict(), "sentinel_pid": pid, "sentinel_restarted": pid != first_pid,
                  "redis_clients": client.info("clients")["connected_clients"],
                  "false_alarms": len(alarms), **(process_stats(pid) if pid else {})}
        with open(out, "a") as f:
            f.write(json.dumps(sample) + "\n")
        print(json.dumps(sample), flush=True)

    summary = {"hours": args.hours, "false_alarms": [
        {"id": s["id"], "kind": s.get("kind"), "node": s.get("node"), "summary": s.get("summary"),
         "closed": s.get("status")} for s in alarms.values()]}
    with open(out, "a") as f:
        f.write(json.dumps({"summary": summary}) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
