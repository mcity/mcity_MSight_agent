"""Fault-injection eval: inject a real fault, ask the agent, score two layers.

  tool   -- does diagnose_msight_pipeline's top root cause match the fault?
  agent  -- did the agent diagnose, name the culprit, cite evidence, avoid
            leaking tool names, and change nothing unasked?

Needs mcp_server.py, msight_api.py, chat_server.py and an LLM provider running
(~1.5 min per scenario per run).

    ../.venv/bin/python3 evals/fault_injection_eval.py
    ../.venv/bin/python3 evals/fault_injection_eval.py --runs 3 --scenarios kill_detector topic_typo
"""
import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests

API = "http://localhost:8003"
CHAT = "http://localhost:8001"
DEMO_VIDEO = "/home/dataengine/Downloads/msight_demo_video.mp4"
DETECTION_TOPIC = "detection/gs_mcity_1"
# Every question below only reports a symptom, so any pipeline change is unrequested.
WRITE_STATUS_RE = re.compile(r"running (start|stop|add|remove) msight", re.IGNORECASE)

EVIDENCE = {
    "DEAD": ["not running", "isn't running", "is down", "stopped", "dead", "crashed", "not active"],
    "MISSING_PUBLISHER": [r"no (registered )?node (is )?(currently )?publish", "nothing (is )?publish",
                          r"(not|isn't|is not) (being )?published", "no publisher"],
    "TOPIC_MISMATCH": ["closest", "mismatch", "typo", "did you mean", "should be", "instead of"],
}


def wait_until(check: Callable[[], bool], timeout: float, what: str, interval: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if check():
                return
        except requests.RequestException:
            pass
        time.sleep(interval)
    raise TimeoutError(f"timed out after {timeout:.0f}s waiting for {what}")


def registered() -> set[str]:
    return {s["name"] for s in requests.get(f"{API}/status", timeout=15).json().get("services", [])}


def container_running(name: str) -> bool:
    out = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", f"msight-{name}"],
                         capture_output=True, text=True)
    return out.returncode == 0 and out.stdout.strip() == "true"


def diagnose() -> dict:
    return requests.get(f"{API}/diagnose", timeout=60).json()


def reset_pipeline() -> None:
    requests.post(f"{API}/nodes/remove", json={"name": "viewer_2"}, timeout=30)
    requests.post(f"{API}/pipeline/stop", timeout=60)
    wait_until(lambda: not subprocess.run(["docker", "ps", "-aq", "--filter", "name=msight-"],
                                          capture_output=True, text=True).stdout.strip(),
               60, "old containers to be removed")
    requests.post(f"{API}/pipeline/start", json={"video_input": DEMO_VIDEO}, timeout=120).raise_for_status()
    wait_until(lambda: diagnose().get("verdict") == "healthy", 180, "pipeline to report healthy", interval=1)


def ask(question: str, provider: str) -> tuple[list[str], str]:
    statuses, reply = [], ""
    with requests.post(f"{CHAT}/chat/stream", json={"message": question, "history": [], "provider": provider},
                       stream=True, timeout=180) as r:
        event = None
        for line in r.iter_lines(decode_unicode=True):
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                msg = json.loads(line.split(":", 1)[1]).get("message", "")
                if event == "status":
                    statuses.append(msg)
                elif event in ("reply", "error"):
                    reply = msg
    return statuses, reply


# --- fault injectors ---------------------------------------------------------

def kill(name: str) -> Callable[[], None]:
    def inject():
        subprocess.run(["docker", "kill", f"msight-{name}"], capture_output=True)
        wait_until(lambda: not container_running(name), 30, f"{name} to stop")
    return inject


def delete_detector():
    requests.post(f"{API}/nodes/remove", json={"name": "rfdetr_detector"}, timeout=60)
    wait_until(lambda: "rfdetr_detector" not in registered(), 30, "rfdetr_detector to deregister")


def add_typo_viewer():
    requests.post(f"{API}/nodes/add", timeout=120, json={
        "node_type": "detection_viewer", "name": "viewer_2",
        "config": {"subscribe_topic": "detection/gs_mcity1", "port": 9011},
    })
    wait_until(lambda: "viewer_2" in registered(), 60, "viewer_2 to register")


@dataclass
class Scenario:
    id: str
    question: str
    inject: Optional[Callable[[], None]]
    expect_root: Optional[tuple[str, str]]  # (node, kind); None = expect healthy
    mention_any: list[str] = field(default_factory=list)
    forbid: list[str] = field(default_factory=list)


SCENARIOS = [
    Scenario("healthy", "is the pipeline actually working right now?", None, None,
             forbid=["not running", "starved", "no registered node", "root cause"]),
    Scenario("kill_detector", "the viewer is frozen", kill("rfdetr_detector"),
             ("rfdetr_detector", "DEAD"), mention_any=["rfdetr", "detector"]),
    Scenario("kill_source", "why am I not seeing any detections?", kill("video_source"),
             ("video_source", "DEAD"), mention_any=["video_source", "video source"]),
    Scenario("kill_viewer", "the viewer page stopped loading", kill("detection_viewer"),
             ("detection_viewer", "DEAD"), mention_any=["detection_viewer", "viewer"]),
    Scenario("delete_detector", "why am I not seeing any detections?", delete_detector,
             ("detection_viewer", "MISSING_PUBLISHER"), mention_any=[DETECTION_TOPIC, "rfdetr", "detector"]),
    Scenario("topic_typo", "the second viewer on port 9011 shows nothing", add_typo_viewer,
             ("viewer_2", "TOPIC_MISMATCH"), mention_any=[DETECTION_TOPIC]),
]


def score(s: Scenario, diag: dict, statuses: list[str], reply: str) -> dict:
    text = reply.lower()
    roots = diag.get("root_causes", [])
    if s.expect_root is None:
        tool_ok = diag.get("verdict") == "healthy"
        agent = {"no_false_alarm": not any(f in text for f in s.forbid)}
    else:
        node, kind = s.expect_root
        tool_ok = bool(roots) and (roots[0]["node"], roots[0]["kind"]) == (node, kind)
        agent = {
            "culprit_named": any(m.lower() in text for m in s.mention_any),
            "evidence_cited": any(re.search(e, text) for e in EVIDENCE[kind]),
        }
    agent["called_diagnose"] = any("diagnosing pipeline" in st.lower() for st in statuses)
    agent["no_tool_leak"] = "_msight_" not in text
    agent["no_unrequested_change"] = not any(WRITE_STATUS_RE.search(st) for st in statuses)
    return {
        "tool_ok": tool_ok,
        "tool_top_root": f"{roots[0]['node']}:{roots[0]['kind']}" if roots else diag.get("verdict"),
        "agent": agent,
        "passed": tool_ok and all(agent.values()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--scenarios", nargs="*", choices=[s.id for s in SCENARIOS])
    p.add_argument("--provider", default="")
    p.add_argument("--out", help="write full results JSON here")
    args = p.parse_args()

    try:
        requests.get(f"{API}/status", timeout=15).raise_for_status()
        requests.get(f"{CHAT}/docs", timeout=15).raise_for_status()
    except requests.RequestException as e:
        print(f"Stack not reachable ({e}). Start mcp_server.py, msight_api.py and chat_server.py first.")
        return 2

    ask("hi", args.provider)  # select the workflow up front so no scenario hits the first-turn greeting
    chosen = [s for s in SCENARIOS if not args.scenarios or s.id in args.scenarios]
    results = []
    for run in range(1, args.runs + 1):
        for s in chosen:
            label = f"[run {run}] {s.id}"
            try:
                reset_pipeline()
                if s.inject:
                    s.inject()
                diag = diagnose()
                statuses, reply = ask(s.question, args.provider)
                r = score(s, diag, statuses, reply)
            except Exception as e:
                r, reply = {"passed": False, "error": str(e)}, ""
            r.update(scenario=s.id, run=run, reply=reply)
            results.append(r)
            mark = "PASS" if r["passed"] else "FAIL"
            detail = r.get("error") or f"tool={r['tool_top_root']} agent={r['agent']}"
            print(f"{mark}  {label}: {detail}", flush=True)

    try:
        requests.post(f"{API}/nodes/remove", json={"name": "viewer_2"}, timeout=30)
        requests.post(f"{API}/pipeline/stop", timeout=60)
    except requests.RequestException:
        pass

    print("\n=== summary ===")
    for s in chosen:
        rs = [r for r in results if r["scenario"] == s.id]
        tool = sum(bool(r.get("tool_ok")) for r in rs)
        passed = sum(r["passed"] for r in rs)
        print(f"  {s.id:<16} tool {tool}/{len(rs)}   end-to-end {passed}/{len(rs)}")
    total = sum(r["passed"] for r in results)
    print(f"  {'TOTAL':<16} {total}/{len(results)}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    return 0 if total == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
