"""End-to-end test of the sentinel: inject real faults into a live stack and score what the
incident feed shows (the same feed the chat page renders).

Per scenario: restore a clean baseline, inject, then measure from the feed
  detection    an incident of the expected kind, on the expected node, and how fast
  diagnosis    Claude's report arrived; requires_human matches what the fault needs
  outcome      with `--autonomy fix`: fixed when a restart fixes it, escalated when not. Never
               fixed when a person was needed; escalating because the diagnosis was unsure
               (low confidence) counts as cautious, not as a miss. Claude's requires_human
               label is scored separately (label_ok).
  recovery     the incident closed (resolved), and how long after the fault
  noise        extra incidents opened by the same fault
A no-fault control checks for false alarms.

Needs mcp_server, msight_api, chat_server and the sentinel running (start the sentinel with
the autonomy level to test). `--setup` starts the camera simulator and puts the pipeline on it.

    PYTHONPATH=. .venv/bin/python3 -m msight_agent.evals.sentinel_e2e --setup
    PYTHONPATH=. .venv/bin/python3 -m msight_agent.evals.sentinel_e2e --scenarios kill_detector camera_frozen --repeat 3
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

import redis

from msight_agent.msight.runtime import incidents
from msight_agent.sentinel import inject
from msight_agent.shared.paths import LOG_DIR, NODE_LOG_DIR, REPO_ROOT

API = "http://localhost:8003"
CAMERA_URL = "rtsp://127.0.0.1:8555/cam"
SENTINEL_PY = str(REPO_ROOT / ".venv-sentinel" / "bin" / "python")
MAIN_PY = str(REPO_ROOT / ".venv" / "bin" / "python3")
OUT_DIR = LOG_DIR / "sentinel_e2e"
SERVICES = {
    "chat_server": (8001, [MAIN_PY, "-m", "uvicorn", "msight_agent.chat_server:app", "--port", "8001"]),
    "msight_api": (8003, [MAIN_PY, "-m", "msight_agent.msight_api"]),
}


# --- plumbing ---------------------------------------------------------------------------

def sh(*argv: str, check: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(list(argv), capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"{' '.join(argv)}: {result.stderr.strip() or result.stdout.strip()}")
    return result


def camera(*args: str) -> None:
    sh(SENTINEL_PY, "-m", "msight_agent.sentinel.camera_sim", *args, check=True)


def api(method: str, path: str, body: Optional[dict] = None, timeout: float = 300) -> dict:
    request = urllib.request.Request(API + path, method=method, headers={"Content-Type": "application/json"},
                                     data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return json.loads(e.read() or b"{}")
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        return {"status": "error", "message": str(e)}


def wait_for(check: Callable[[], bool], timeout: float, interval: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if check():
                return True
        except Exception:  # noqa: BLE001 -- keep polling through transient errors
            pass
        time.sleep(interval)
    return False


def port_pid(port: int) -> Optional[int]:
    out = sh("lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN").stdout.split()
    return int(out[0]) if out else None


def start_service(name: str) -> None:
    port, cmd = SERVICES[name]
    if port_pid(port):
        return
    (OUT_DIR / "services").mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "services" / f"{name}.log", "ab") as log:
        subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                         env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
    wait_for(lambda: port_pid(port) is not None, 60, 1)


# --- the incident feed ------------------------------------------------------------------

class Feed:
    """Follows MSIGHT:INCIDENTS and folds it into incident states."""

    def __init__(self):
        self.redis = redis.Redis(socket_timeout=10)
        self.states: dict[str, dict] = {}
        recent = incidents.read_recent(self.redis)
        self.states = incidents.fold(e for _, e in recent)
        self.cursor = recent[-1][0] if recent else "0-0"

    def poll(self, block_ms: int = 1000) -> None:
        result = self.redis.xread({incidents.STREAM_KEY: self.cursor}, count=200, block=block_ms)
        for stream_id, fields in (result[0][1] if result else []):
            self.cursor = stream_id.decode() if isinstance(stream_id, bytes) else stream_id
            event = incidents.decode(fields)
            if event and event.get("incident"):
                self.states[event["incident"]] = incidents.apply(self.states.get(event["incident"]), event)

    def since(self, t0: float) -> list[dict]:
        return sorted((s for s in self.states.values() if s["detected_at"] >= t0 and not s["id"].startswith("FEED-")),
                      key=lambda s: s["detected_at"])

    def open_count(self) -> int:
        return sum(1 for s in self.states.values() if s["status"] == "open" and not s["id"].startswith("FEED-"))


# --- scenarios ----------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    category: str
    inject: Callable[[], None]
    kinds: tuple[str, ...]            # acceptable kinds for the primary incident
    node: Optional[str]               # expected node (substring of the incident's node field)
    human: Optional[bool]             # expected requires_human (None: either is fine)
    fixable: bool = False             # a scoped restart fixes it (autonomy fix should fix, not escalate)
    detect_timeout: float = 90
    camera_only: bool = False
    note: str = ""
    restore: Optional[Callable[[], None]] = None   # extra cleanup beyond the baseline


def _kill_service_then_restart(name: str, after: float = 40) -> None:
    pid = port_pid(SERVICES[name][0])
    if pid:
        os.kill(pid, 9)
    threading.Timer(after, start_service, args=(name,)).start()


def _kill_recording_process() -> None:
    api("POST", "/recording/start", {})
    wait_for(lambda: (NODE_LOG_DIR / "local_dumper.pid").is_file(), 60)
    time.sleep(10)
    inject.kill_process("local_dumper")


def _add_broken_detector() -> None:
    api("POST", "/nodes/add", {"node_type": "rfdetr_detector", "name": "detector_2", "config": {
        "publish_topic": "detection/test_2", "subscribe_topic": "camera/gs_mcity_1",
        "det_configs": "/configs/does_not_exist.yaml", "sensor_name": "gs_mcity_1"}})


SCENARIOS = [
    Scenario("kill_detector", "node", lambda: inject.kill("rfdetr_detector"),
             ("CONTAINER_KILLED",), "rfdetr_detector", False, fixable=True),
    Scenario("oom_detector", "node",
             lambda: inject.oom("rfdetr_detector", inject.OOM_LIMIT),
             ("CONTAINER_OOM",), "rfdetr_detector", None, fixable=True, detect_timeout=120),
    Scenario("exit_detector", "node", lambda: inject.exit_cleanly("rfdetr_detector"),
             ("CONTAINER_EXITED",), "rfdetr_detector", False, fixable=True),
    Scenario("freeze_detector", "node", lambda: inject.freeze("rfdetr_detector"),
             ("NODE_UNRESPONSIVE", "PIPELINE_DEGRADED"), "rfdetr_detector", False, fixable=True),
    Scenario("broken_config_node", "node", _add_broken_detector, ("CONTAINER_CRASHED",), "detector_2", True,
             detect_timeout=180, note="crash loop from a bad config on an added node",
             restore=lambda: api("POST", "/nodes/remove", {"name": "detector_2"})),
    Scenario("kill_recording_process", "node", _kill_recording_process, ("PROCESS_DIED",), "local_dumper", None,
             fixable=True, note="host-process node", restore=lambda: api("POST", "/recording/stop", {})),
    Scenario("chat_server_crash", "service", lambda: _kill_service_then_restart("chat_server"),
             ("SERVICE_DOWN",), "chat_server", None, detect_timeout=60),
    Scenario("msight_api_crash", "service", lambda: _kill_service_then_restart("msight_api"),
             ("SERVICE_DOWN",), "msight_api", None, detect_timeout=60),
    Scenario("camera_offline", "stream", lambda: camera("network", "offline"),
             ("CONTAINER_CRASHED", "CONTAINER_EXITED", "NODE_UNRESPONSIVE"), "video_source", True,
             detect_timeout=120, camera_only=True),
    Scenario("camera_reboot", "stream", lambda: camera("camera", "reboot"),
             ("CONTAINER_CRASHED", "CONTAINER_EXITED", "NODE_UNRESPONSIVE"), "video_source", None,
             fixable=True, detect_timeout=120, camera_only=True, note="brief outage; may recover on its own"),
    Scenario("network_stall", "stream", lambda: camera("network", "stall"),
             ("NODE_UNRESPONSIVE", "PIPELINE_DEGRADED", "CONTAINER_CRASHED", "CAMERA_FROZEN", "CAMERA_LOW_FPS"),
             "video_source", True,
             detect_timeout=120, camera_only=True),
    Scenario("network_reset", "stream", lambda: camera("network", "reset"),
             ("CONTAINER_CRASHED", "CONTAINER_EXITED", "NODE_UNRESPONSIVE"), "video_source", True,
             detect_timeout=120, camera_only=True),
    Scenario("network_slow", "stream", lambda: camera("network", "slow"),
             ("CAMERA_LOW_FPS", "NODE_UNRESPONSIVE", "PIPELINE_DEGRADED", "CONTAINER_CRASHED"), "video_source", True,
             detect_timeout=180, camera_only=True),
    Scenario("camera_credentials", "stream", lambda: camera("auth", "on"),
             ("CONTAINER_CRASHED", "CONTAINER_EXITED", "NODE_UNRESPONSIVE"), "video_source", True,
             detect_timeout=120, camera_only=True),
    Scenario("camera_frozen", "picture", lambda: camera("picture", "frozen"),
             ("CAMERA_FROZEN",), "video_source", True, camera_only=True),
    Scenario("camera_black", "picture", lambda: camera("picture", "black"),
             ("CAMERA_BLACK",), "video_source", True, camera_only=True),
    Scenario("camera_covered", "picture", lambda: camera("picture", "covered"),
             ("CAMERA_COVERED",), "video_source", True, camera_only=True),
    Scenario("camera_blurred", "picture", lambda: camera("picture", "blur"),
             ("CAMERA_DEFOCUSED",), "video_source", True, camera_only=True,
             note="can also crash the detector (MSight_Vision localizer bounds bug)"),
    Scenario("camera_moved", "picture", lambda: camera("picture", "moved"),
             ("CAMERA_MOVED",), "video_source", True, camera_only=True),
    Scenario("no_fault", "control", lambda: None, (), None, None, detect_timeout=120),
]


# --- one run --------------------------------------------------------------------------------

def baseline(source: dict, feed: Feed, camera_mode: bool) -> bool:
    """Healthy pipeline, normal camera, no extra nodes, no open incidents."""
    for name in SERVICES:
        start_service(name)
    if camera_mode:
        camera("restore")
    for name in sh("docker", "ps", "-a", "--format", "{{.Names}}", "--filter", "name=msight-").stdout.split():
        info = json.loads(sh("docker", "inspect", name).stdout or "[{}]")[0]
        if info.get("State", {}).get("Paused"):
            sh("docker", "unpause", name)
        if (info.get("HostConfig", {}).get("Memory") or 0) > 0:
            sh("docker", "rm", "-f", name)
    api("POST", "/nodes/remove", {"name": "detector_2"})
    api("POST", "/recording/stop", {})
    api("POST", "/pipeline/start", source)
    healthy = wait_for(lambda: api("GET", "/diagnose", timeout=60).get("verdict") == "healthy", 240, 3)

    def quiet() -> bool:
        feed.poll(500)
        return feed.open_count() == 0
    return healthy and wait_for(quiet, 240, 1)


def run(scenario: Scenario, source: dict, feed: Feed, camera_mode: bool, settle: float) -> dict:
    result = {"scenario": scenario.name, "group": scenario.category, "started": time.strftime("%H:%M:%S")}
    result["baseline_ok"] = baseline(source, feed, camera_mode)
    time.sleep(settle)                           # let the sentinel re-learn rhythms after the restore
    t0 = time.time()
    scenario.inject()

    if not scenario.kinds:                       # no-fault control: nothing may open
        end = time.monotonic() + scenario.detect_timeout
        while time.monotonic() < end:
            feed.poll(1000)
        opened = feed.since(t0)
        result.update(incidents=[s["kind"] for s in opened], passed=not opened)
        return result

    def on_target(state: dict) -> bool:
        # Settled (past the provisional "detected" stage) and not withdrawn by the sentinel itself:
        # a starved or merely slow node is dismissed and the real cause reported instead.
        return ((scenario.node is None or scenario.node in (state.get("node") or ""))
                and state["stage"] != "detected" and state["status"] != "dismissed")

    primary = None
    end = time.monotonic() + scenario.detect_timeout
    while time.monotonic() < end and (primary is None or not on_target(primary)):
        feed.poll(1000)
        matching = [s for s in feed.since(t0) if s.get("kind") in scenario.kinds]
        primary = next((s for s in matching if on_target(s)), matching[0] if matching else None)
    opened = feed.since(t0)
    result.update(detected=primary is not None, incidents=[f"{s.get('kind')}:{s.get('node')}" for s in opened])
    if primary is None:
        result["passed"] = False
        return result

    pid = primary["id"]
    result.update(kind=primary["kind"], node=primary.get("node"), detect_s=round(primary["detected_at"] - t0, 1),
                  node_ok=scenario.node is None or scenario.node in (primary.get("node") or ""))

    def diagnosed() -> bool:
        # Not "or closed": a service that comes back on its own closes before Claude reports.
        feed.poll(1000)
        state = feed.states[pid]
        return bool(state.get("report") or state.get("error"))
    wait_for(diagnosed, 300, 0)
    state = feed.states[pid]
    report = state.get("report") or {}
    result.update(diagnosed=bool(report), category=report.get("category"), confidence=report.get("confidence"),
                  requires_human=report.get("requires_human"), checks=len(state.get("checks") or []),
                  cost_usd=state.get("cost_usd"), autonomy=state.get("autonomy"))
    # Claude's own label, scored separately: pass/fail judges what the system then did.
    result["label_ok"] = scenario.human is None or report.get("requires_human") == scenario.human
    result["human_ok"] = result["label_ok"]

    if state.get("autonomy") == "fix":           # wait for the fixer's verdict
        wait_for(lambda: (feed.poll(1000), feed.states[pid].get("fix_outcome"))[1] is not None
                 or feed.states[pid]["stage"] == "escalated", 300, 0)
        state = feed.states[pid]
        result["fix_outcome"] = state.get("fix_outcome") or ("escalated" if state["stage"] == "escalated" else None)
        result["fix_ok"] = (result["fix_outcome"] == "fixed") == scenario.fixable or (
            scenario.fixable and state["status"] == "resolved" and result["fix_outcome"] is None)
        result["escalation"] = state.get("fix_text") if result["fix_outcome"] == "escalated" else None
        # The path, not just the outcome: every action that didn't end in a verified fix was wasted.
        result["actions"] = list(state.get("fix_actions") or [])
        result["wasted_actions"] = len(result["actions"]) - (1 if result["fix_outcome"] == "fixed" else 0)
        # Fixing what needed a person is the failure. Escalating because the diagnosis was
        # unsure is cautious, not a miss: low confidence goes to a person by design.
        cautious = result["fix_outcome"] == "escalated" and "low confidence" in (result["escalation"] or "")
        result["cautious"] = cautious
        result["human_ok"] = not (scenario.human and result["fix_outcome"] == "fixed")
        result["fix_ok"] = result["fix_ok"] or cautious

    if state["status"] == "open" and not (state.get("autonomy") == "fix" and result.get("fix_outcome") == "fixed"):
        if scenario.restore:
            scenario.restore()
        if camera_mode:
            camera("restore")                    # human-needed faults: restore, then the incident must close
        api("POST", "/pipeline/start", source)
    closed = wait_for(lambda: (feed.poll(1000), feed.states[pid]["status"] != "open")[1], 240, 0)
    state = feed.states[pid]
    closed_at = state.get("resolved_at") or state.get("dismissed_at")
    result.update(closed=state["status"] if closed else "still open",
                  recover_s=round(closed_at - t0, 1) if closed_at else None,
                  extra_incidents=len(feed.since(t0)) - 1)
    if scenario.restore:
        scenario.restore()
    result["passed"] = all([result["detected"], result["node_ok"], result["diagnosed"], result["human_ok"],
                            result.get("fix_ok", True), state["status"] == "resolved"])
    return result


def table(results: list[dict]) -> str:
    cols = ["scenario", "detect_s", "kind", "category", "confidence", "requires_human", "label_ok", "fix_outcome",
            "cautious", "recover_s", "extra_incidents", "passed"]
    rows = [[str(r.get(c, "")) for c in cols] for r in results]
    widths = [max(len(c), *(len(row[i]) for row in rows)) for i, c in enumerate(cols)]
    line = lambda cells: "  ".join(cell.ljust(w) for cell, w in zip(cells, widths))  # noqa: E731
    return "\n".join([line(cols), line(["-" * w for w in widths]), *(line(r) for r in rows)])


# --- the summary: rates with intervals, reliability, per stage ------------------------------

def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a rate: honest at small n and at 0% or 100%."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _rate(flags: list[bool]) -> str:
    k, n = sum(flags), len(flags)
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%}  [95% CI {lo:.0%}-{hi:.0%}]"


STAGES = [  # (label, which results count, did that stage succeed)
    ("detected", lambda r: "detected" in r, lambda r: bool(r.get("detected"))),
    ("right node", lambda r: r.get("detected"), lambda r: bool(r.get("node_ok"))),
    ("diagnosed", lambda r: r.get("detected"), lambda r: bool(r.get("diagnosed"))),
    ("needs-a-person label right", lambda r: r.get("diagnosed"), lambda r: bool(r.get("label_ok"))),
    ("never fixed what needed a person", lambda r: "fix_outcome" in r, lambda r: bool(r.get("human_ok"))),
    ("fixed when fixable (or cautious)", lambda r: "fix_outcome" in r, lambda r: bool(r.get("fix_ok"))),
    ("recovered (watcher verified)", lambda r: r.get("detected"), lambda r: r.get("closed") == "resolved"),
]


def summarize(results: list[dict]) -> dict:
    """What a reviewer asks for: the pass rate with its interval, reliability over repeats
    (pass^k: every run passed), each stage on its own, the path, cost per success, and
    whether stated confidence means anything."""
    runs = [r for r in results if "scenario" in r]
    by_scenario: dict[str, list[dict]] = {}
    for r in runs:
        by_scenario.setdefault(r["scenario"], []).append(r)
    reps = max((len(v) for v in by_scenario.values()), default=0)
    passed = [bool(r.get("passed")) for r in runs]
    correct = sum(passed)
    cost = sum(r.get("cost_usd") or 0 for r in runs)
    confidence: dict[str, list[bool]] = {}
    for r in runs:
        if r.get("diagnosed"):
            confidence.setdefault(r.get("confidence") or "?", []).append(bool(r.get("label_ok")))
    return {
        "runs": len(runs), "scenarios": len(by_scenario), "repeats": reps,
        "pass_rate": _rate(passed),
        "pass_at_k": _rate([any(r.get("passed") for r in v) for v in by_scenario.values()]),
        "pass_hat_k": _rate([all(r.get("passed") for r in v) for v in by_scenario.values()]),
        "flaky": sorted(name for name, v in by_scenario.items() if len({bool(r.get("passed")) for r in v}) > 1),
        "stages": {label: _rate([ok(r) for r in runs if counts(r)]) for label, counts, ok in STAGES},
        "actions": sum(len(r.get("actions") or []) for r in runs),
        "wasted_actions": sum(max(0, r.get("wasted_actions") or 0) for r in runs),
        "cost_usd": round(cost, 2), "cost_per_pass_usd": round(cost / correct, 3) if correct else None,
        "label_right_by_confidence": {level: _rate(flags) for level, flags in sorted(confidence.items())},
    }


def summary_text(summary: dict) -> str:
    lines = [f"{summary['runs']} runs of {summary['scenarios']} scenarios ({summary['repeats']} repeat(s))",
             f"  pass rate          {summary['pass_rate']}",
             f"  pass@k (any run)   {summary['pass_at_k']}",
             f"  pass^k (every run) {summary['pass_hat_k']}"]
    if summary["flaky"]:
        lines.append(f"  flaky              {', '.join(summary['flaky'])}")
    lines.append("  by stage:")
    lines += [f"    {label:<34} {rate}" for label, rate in summary["stages"].items()]
    lines.append(f"  path: {summary['actions']} fix actions, {summary['wasted_actions']} that didn't end in a fix")
    lines.append(f"  cost: ${summary['cost_usd']} total, ${summary['cost_per_pass_usd']} per passed run")
    lines.append("  needs-a-person label right, by Claude's stated confidence:")
    lines += [f"    {level:<8} {rate}" for level, rate in summary["label_right_by_confidence"].items()]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="*", help="names to run (default: all)")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--mode", choices=("camera", "file"), default="camera",
                        help="camera: pipeline on the RTSP simulator (all scenarios); file: the demo video")
    parser.add_argument("--video", default=os.environ.get("MSIGHT_DEMO_VIDEO_PATH"))
    parser.add_argument("--setup", action="store_true", help="start the camera simulator first")
    parser.add_argument("--settle", type=float, default=20, help="seconds after the baseline before injecting")
    parser.add_argument("--summarize", metavar="RESULTS_JSONL", help="only print the summary of an earlier run")
    args = parser.parse_args()

    if args.summarize:
        with open(args.summarize) as f:
            print(summary_text(summarize([json.loads(line) for line in f if line.strip()])))
        return 0

    if not redis.Redis().exists(incidents.SENTINEL_HEARTBEAT_KEY):
        print("The sentinel isn't running (no heartbeat in Redis). Start it first.", file=sys.stderr)
        return 2
    camera_mode = args.mode == "camera"
    if camera_mode and args.setup:
        camera("up", "--video", args.video)
    source = {"rtsp_url": CAMERA_URL} if camera_mode else {"video_input": args.video}
    chosen = [s for s in SCENARIOS if (not args.scenarios or s.name in args.scenarios)
              and (camera_mode or not s.camera_only)]
    feed = Feed()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"run-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    results = []
    for repetition in range(args.repeat):
        for scenario in chosen:
            print(f"[{time.strftime('%H:%M:%S')}] {scenario.name} (run {repetition + 1}/{args.repeat})", flush=True)
            try:
                result = run(scenario, source, feed, camera_mode, args.settle)
            except Exception as e:  # noqa: BLE001 -- record and move on
                result = {"scenario": scenario.name, "passed": False, "error": f"{type(e).__name__}: {e}"}
            result["repetition"] = repetition + 1
            results.append(result)
            with open(out, "a") as f:
                f.write(json.dumps(result, default=str) + "\n")
            print(f"    -> {'PASS' if result.get('passed') else 'FAIL'}  {json.dumps(result, default=str)}", flush=True)
    baseline(source, feed, camera_mode)
    print("\n" + table(results))
    summary = summarize(results)
    with open(out, "a") as f:
        f.write(json.dumps({"summary": summary}) + "\n")
    print("\n" + summary_text(summary))
    passed = sum(1 for r in results if r.get("passed"))
    print(f"\n{passed}/{len(results)} passed. Results: {out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
