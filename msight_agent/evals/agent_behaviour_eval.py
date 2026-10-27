"""Agent-behaviour eval: real conversations against the live stack, scored on what
actually happened (nodes, topics, ports, files) -- never on the reply's wording alone.

Each scenario starts from the same state: a fresh chat session and the demo pipeline
running healthy, with any extra nodes it sets up. The scenarios are the ones found by
hand while testing the node catalog and the stop guard, so a fixed bug stays fixed and
models can be compared:

    ../.venv/bin/python3 evals/agent_behaviour_eval.py
    ../.venv/bin/python3 evals/agent_behaviour_eval.py --provider claude --runs 3
    ../.venv/bin/python3 evals/agent_behaviour_eval.py --scenarios dup_detector thanks_no_stop

Needs mcp_server.py, msight_api.py and chat_server.py running, plus an LLM provider
(~1-2 min per scenario: every scenario restarts the demo pipeline).
"""
import argparse
import json
import re
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import requests

from msight_agent.agent.state import MsightPipelineState, WorkflowState
from msight_agent.agent.tool_schema import tools as TOOL_SCHEMA
from msight_agent.msight.runtime.control_plane import MSightControlPlane
from msight_agent.shared.paths import DATA_DIR, MSIGHT_UPLOADS_DIR

API = "http://localhost:8003"
CHAT = "http://localhost:8001"
DEMO_VIDEO = "/home/dataengine/Downloads/msight_demo_video.mp4"
FIXED = ("video_source", "rfdetr_detector", "detection_viewer")
CAMERA, DETECTIONS = "camera/gs_mcity_1", "detection/gs_mcity_1"
SAMPLE_IMAGE = "/home/dataengine/Github/MSight_Vision/examples/fullstack/basemap_configs/isc.png"
CLIPS_DIR = DATA_DIR / "eval_clips"
SETTLE_SECONDS = 12  # RF-DETR loads its model before it registers
# The agent's real tool names (a pattern also matched paths like ".../mcity_MSight_agent/...").
TOOL_NAME_RE = re.compile(r"\b(" + "|".join(t["function"]["name"] for t in TOOL_SCHEMA) + r")\b")


# --- live state ----------------------------------------------------------------

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


def live() -> dict[str, dict]:
    """{name: {"publish": [...], "subscribe": [...], "url": ...}} for every running node."""
    status = requests.get(f"{API}/status", timeout=30).json()
    urls = status.get("viewer_urls", {})
    as_list = lambda v: [v] if isinstance(v, str) else list(v or [])  # noqa: E731
    return {s["name"]: {"publish": as_list(s.get("publish_topic")), "subscribe": as_list(s.get("subscribe_topic")),
                        "url": urls.get(s["name"])}
            for s in status.get("services", []) if s.get("alive")}


def node_type(name: str) -> Optional[str]:
    import asyncio
    return asyncio.run(MSightControlPlane().node_type_of(name))


def publishers(nodes: dict, topic: str) -> list[str]:
    return [n for n, v in nodes.items() if topic in v["publish"]]


def port_listening(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def remove_all_nodes() -> None:
    requests.post(f"{API}/pipeline/stop", timeout=60)
    for name in live():
        requests.post(f"{API}/nodes/remove", json={"name": name}, timeout=60)
    wait_until(lambda: not live(), 60, "every node to stop")


def add_node(node_type_: str, name: str, config: dict) -> None:
    r = requests.post(f"{API}/nodes/add", json={"node_type": node_type_, "name": name, "config": config}, timeout=180)
    if r.json().get("status") != "ok":
        raise RuntimeError(f"setup could not add {name}: {r.json().get('message')}")


def fresh_session() -> None:
    """A new chat session that already knows the demo is running (as after "run demo")."""
    WorkflowState(workflow_name="msight_pipeline", msight_pipeline=MsightPipelineState(
        mode="demo", video_input=DEMO_VIDEO, pipeline_running=True)).save()


def start_demo() -> None:
    remove_all_nodes()
    requests.post(f"{API}/pipeline/start", json={"video_input": DEMO_VIDEO}, timeout=180).raise_for_status()
    wait_until(lambda: requests.get(f"{API}/diagnose", timeout=60).json().get("verdict") == "healthy",
               180, "the demo pipeline to report healthy", interval=2)


def chat(message: str, history: list, provider: str) -> str:
    reply = ""
    with requests.post(f"{CHAT}/chat/stream", json={"message": message, "history": history, "provider": provider},
                       stream=True, timeout=600) as r:
        event = None
        for line in r.iter_lines(decode_unicode=True):
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and event in ("reply", "error"):
                reply = json.loads(line.split(":", 1)[1]).get("message", "")
    history.append([message, reply])
    return reply


# --- scenarios -------------------------------------------------------------------

@dataclass
class Run:
    before: dict
    after: dict
    replies: list[str]

    @property
    def added(self) -> list[str]:
        return sorted(set(self.after) - set(self.before))

    @property
    def text(self) -> str:
        return "\n".join(self.replies).lower()


@dataclass
class Scenario:
    id: str
    messages: list[str]
    check: Callable[[Run], dict[str, bool]]
    setup: Optional[Callable[[], None]] = None
    wait_for: Optional[Callable[[dict], bool]] = None  # extra settle condition after the last message
    tags: list[str] = field(default_factory=list)


def extra_image_source() -> None:
    add_node("local_image", "img_test", {"image_path": SAMPLE_IMAGE, "publish_topic": "camera/img_test",
                                         "sensor_name": "img_test", "fps": 5})


def three_extras() -> None:
    extra_image_source()
    add_node("detection_viewer", "viewer_extra", {"subscribe_topic": DETECTIONS})
    add_node("frame_annotator", "annotator_extra", {"subscribe_topic": DETECTIONS, "publish_topic": "annotated/extra"})


def fixed_alive(run: Run) -> bool:
    return all(n in run.after for n in FIXED)


def check_mp4_source(run: Run) -> dict:
    new = [n for n in run.added if node_type(n) == "video_source_mp4_folder"]
    return {
        "added_mp4_folder_node": bool(new),
        "publishes_requested_topic": any("camera/folder_test" in run.after[n]["publish"] for n in new),
        "pipeline_untouched": fixed_alive(run),
    }


def check_no_stop(run: Run) -> dict:
    return {
        "pipeline_still_running": fixed_alive(run),
        "extra_node_still_running": "img_test" in run.after,
        "mentions_running_nodes": "img_test" in run.text or "still running" in run.text,
    }


def check_dup_detector(run: Run) -> dict:
    new = [n for n in run.added if node_type(n) == "rfdetr_detector"]
    used = [t for n in new for t in run.after[n]["publish"]]
    return {
        "single_publisher_on_topic": len(publishers(run.after, DETECTIONS)) == 1,
        # Either refused outright, or a different topic used and the user told which.
        "user_told_topic": not new or any(t.lower() in run.text for t in used),
    }


def check_wrong_kind_viewer(run: Run) -> dict:
    return {
        "no_viewer_on_camera_topic": not any(CAMERA in run.after[n]["subscribe"] for n in run.added),
        "suggests_detection_topic": DETECTIONS in run.text,
    }


def check_unavailable(run: Run) -> dict:
    return {
        "nothing_added": not run.added,
        "says_unavailable": bool(re.search(r"not (a )?(available|supported|offered|recognized|recognised|valid)|"
                                           r"isn't (available|supported|recognized)|no .*bytes_viewer|"
                                           r"unavailable|doesn't exist|does not exist|unknown node|don'?t see|not in the catalog|"
                                           r"don'?t include", run.text)),
    }


def check_vague_detector(run: Run) -> dict:
    new = [n for n in run.added if node_type(n) == "rfdetr_detector"]
    return {
        "one_detector_added": len(new) == 1,
        "own_topic": len(new) == 1 and DETECTIONS not in run.after[new[0]]["publish"],
        "no_internal_settings_asked": "det_configs" not in run.text,
    }


def check_second_viewer(run: Run) -> dict:
    new = [n for n in run.added if node_type(n) == "detection_viewer"]
    ports = [int(run.after[n]["url"].rsplit(":", 1)[1]) for n in new if run.after[n]["url"]]
    return {
        "viewer_added": bool(new),
        "own_port_listening": any(p != 9010 and port_listening(p) for p in ports),
        "real_link_given": any(str(p) in run.text for p in ports),
    }


def check_chain(run: Run) -> dict:
    clips = list(CLIPS_DIR.rglob("*.mp4"))
    return {
        "annotator_aggregator_dumper_added": len(run.added) >= 3,
        "clips_written": bool(clips),
    }


def check_stop_lists_extras(run: Run) -> dict:
    return {
        "pipeline_stopped": not any(n in run.after for n in FIXED),
        "extra_left_running": "img_test" in run.after,
        "user_told_about_extra": "img_test" in run.text,
    }


def check_remove_all(run: Run) -> dict:
    return {"nothing_left_running": not run.after}


def check_remove_additional(run: Run) -> dict:
    return {
        "standard_pipeline_kept": fixed_alive(run),
        "added_nodes_removed": not any(n in run.after for n in ("img_test", "viewer_extra", "annotator_extra")),
    }


def check_detector_on_detections(run: Run) -> dict:
    new = [n for n in run.added if node_type(n) == "rfdetr_detector"]
    return {
        "no_detector_on_detection_topic": not any(DETECTIONS in run.after[n]["subscribe"] for n in new),
        "user_told_why": bool(re.search(r"detection(s)? (data|topic)|frames|image|camera", run.text)),
    }


def check_unknown_key(run: Run) -> dict:
    new = [n for n in run.added if node_type(n) == "local_image"]
    return {"image_source_added": bool(new), "pipeline_untouched": fixed_alive(run)}


def clips_appeared(_after: dict) -> bool:
    return bool(list(CLIPS_DIR.rglob("*.mp4")))


SCENARIOS = [
    Scenario("mp4_source",
             [f"add an mp4 folder source on {MSIGHT_UPLOADS_DIR}, publishing camera/folder_test, sensor folder_test"],
             check_mp4_source, tags=["node_choice"]),
    Scenario("thanks_no_stop", ["thanks, that's all for now"], check_no_stop,
             setup=extra_image_source, tags=["stop_guard"]),
    Scenario("dup_detector", ["add another rfdetr detector publishing to detection/gs_mcity_1"],
             check_dup_detector, tags=["wiring"]),
    Scenario("wrong_kind_viewer", ["add a detection viewer on camera/gs_mcity_1"],
             check_wrong_kind_viewer, tags=["wiring"]),
    Scenario("unavailable_node", ["add a bytes_viewer on camera/gs_mcity_1"], check_unavailable, tags=["catalog"]),
    Scenario("vague_detector", ["add another rfdetr detector"], check_vague_detector, tags=["wiring"]),
    Scenario("second_viewer", ["add a second detection viewer"], check_second_viewer, tags=["ports"]),
    Scenario("chain", [
        f"add a frame annotator on {DETECTIONS} publishing annotated/eval",
        "add a video aggregator on annotated/eval publishing video/eval",
        f"add a video local dumper on video/eval saving to {CLIPS_DIR}",
    ], check_chain, wait_for=clips_appeared, tags=["wiring"]),
    Scenario("stop_lists_extras", ["stop the pipeline"], check_stop_lists_extras,
             setup=extra_image_source, tags=["stop_guard"]),
    Scenario("remove_all", ["stop the pipeline", "remove all the remaining nodes"], check_remove_all,
             setup=three_extras, tags=["cleanup"]),
    Scenario("remove_additional", ["remove all the additional nodes"], check_remove_additional,
             setup=three_extras, tags=["cleanup"]),
    Scenario("detector_on_detections", ["add a second rfdetr detector reading detection/gs_mcity_1"],
             check_detector_on_detections, tags=["wiring"]),
    Scenario("image_source", [f"add a local_image source using {SAMPLE_IMAGE}, publishing camera/img_eval, "
                              "sensor img_eval, at 5 frames per second"], check_unknown_key, tags=["config_keys"]),
]


def run_scenario(s: Scenario, provider: str) -> dict:
    start_demo()
    if s.setup:
        s.setup()
    fresh_session()
    if CLIPS_DIR.exists():
        for f in CLIPS_DIR.rglob("*"):
            if f.is_file():
                f.unlink()
    before = live()
    history: list = []
    replies = [chat(m, history, provider) for m in s.messages]
    time.sleep(SETTLE_SECONDS)
    if s.wait_for:
        try:
            wait_until(lambda: s.wait_for(live()), 45, f"{s.id} to settle")
        except TimeoutError:
            pass  # scored as-is
    result = Run(before=before, after=live(), replies=replies)
    checks = s.check(result)
    checks["no_tool_names_leaked"] = not TOOL_NAME_RE.search(result.text)
    return {"checks": checks, "passed": all(checks.values()), "added": result.added, "replies": replies}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--scenarios", nargs="*", choices=[s.id for s in SCENARIOS])
    p.add_argument("--provider", default="", help="openai | claude | gemini | groq (default: the server's)")
    p.add_argument("--out", help="write full results JSON here")
    args = p.parse_args()

    try:
        requests.get(f"{API}/status", timeout=15).raise_for_status()
        requests.get(f"{CHAT}/docs", timeout=15).raise_for_status()
    except requests.RequestException as e:
        print(f"Stack not reachable ({e}). Start mcp_server.py, msight_api.py and chat_server.py first.")
        return 2

    chosen = [s for s in SCENARIOS if not args.scenarios or s.id in args.scenarios]
    results = []
    for run in range(1, args.runs + 1):
        for s in chosen:
            label = f"[run {run}] {s.id}"
            try:
                r = run_scenario(s, args.provider)
            except Exception as e:
                r = {"passed": False, "error": f"{type(e).__name__}: {e}"}
            r.update(scenario=s.id, run=run)
            results.append(r)
            detail = r.get("error") or ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in r["checks"].items())
            print(f"{'PASS' if r['passed'] else 'FAIL'}  {label}: {detail}", flush=True)

    try:
        remove_all_nodes()
    except Exception:
        pass
    WorkflowState().save()

    print("\n=== summary ===")
    for s in chosen:
        rs = [r for r in results if r["scenario"] == s.id]
        print(f"  {s.id:<20} {sum(r['passed'] for r in rs)}/{len(rs)}")
    total = sum(r["passed"] for r in results)
    print(f"  {'TOTAL':<20} {total}/{len(results)}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
    return 0 if total == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
