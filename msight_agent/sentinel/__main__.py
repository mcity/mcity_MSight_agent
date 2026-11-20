"""python -m msight_agent.sentinel [--autonomy observe|suggest|fix] [--model ...]

Watches the deployment; each incident is announced to the chat feed as soon as it's
seen, investigated by Claude (one at a time, in order) with live progress, and
written to data/logs/incidents/. Autonomy decides what happens after the diagnosis:
observe (report only), suggest (a Fix-it button for a person, the default), or fix
(Claude applies a scoped fix itself and the watcher verifies it)."""
import argparse
import asyncio
import os
import signal
import sys
import time
from collections import defaultdict

from dotenv import dotenv_values

from msight_agent.msight.runtime.utils import redis_kwargs
from msight_agent.sentinel.checks import MAX_FIXES_PER_WINDOW, autofix_decision, secret_env_names
from msight_agent.sentinel.incident_log import INCIDENT_DIR, append_event, write_incident
from msight_agent.sentinel.stream import IncidentPublisher
from msight_agent.sentinel.watcher import Incident, Watcher, clock
from msight_agent.shared.paths import REPO_ROOT

_BOLD, _RED, _RESET = ("\033[1m", "\033[31m", "\033[0m") if sys.stdout.isatty() else ("", "", "")


def say(line: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {line}", flush=True)


VERIFY_SECONDS = 150  # after a fix, how long the watcher gets to see the incident clear


async def wait_until_closed(watcher: Watcher, incident_id: str, seconds: float) -> bool:
    """Whether the watcher closed the incident (health restored) within `seconds`."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if incident_id not in watcher.open:
            return True
        await asyncio.sleep(2)
    return False


async def autofix(incident: Incident, report: dict, warnings: list[str], args: argparse.Namespace,
                  publish: IncidentPublisher, watcher: Watcher, history: dict[str, list[float]]) -> dict:
    """--autonomy fix: decide in code, let Claude act, verify by the watcher. Returns the record."""
    from msight_agent.sentinel.fixer import attempt_fix  # needs claude_agent_sdk
    ok, why_not = autofix_decision(incident.to_dict(), report, warnings, history[incident.node], time.time(),
                                   max_fixes=args.max_fixes)
    if not ok:
        say(f"{_BOLD}ESCALATED (not fixing automatically):{_RESET} {why_not}")
        publish(incident.id, "escalated", text=why_not, human_action=report.get("human_action") if report else None)
        return {"outcome": "escalated", "reason": why_not, "acted": False}

    say(f"{_BOLD}Claude is fixing it...{_RESET}")
    publish(incident.id, "fixing")
    record = await attempt_fix(incident, report, say, lambda text: publish(incident.id, "progress", text=text),
                               model=args.model, mcp_url=args.mcp_url)
    result = record.get("report") or {}
    outcome = result.get("outcome") or "escalated"
    if record["actions"]:
        history[incident.node].append(time.time())
    if outcome == "fixed" or (outcome == "no_action_needed" and incident.id in watcher.open):
        if await wait_until_closed(watcher, incident.id, VERIFY_SECONDS):
            text = f"Fixed automatically: {'; '.join(record['actions']) or 'no action needed'}. {result.get('reason', '')}"
            say(f"{_BOLD}FIXED AND VERIFIED:{_RESET} {text}")
            publish(incident.id, "fixed", text=text, fix_actions=record["actions"])
            return {**record, "outcome": "fixed", "verified": True}
        outcome, result = "escalated", {"reason": f"The fix was applied but the pipeline didn't recover within "
                                                  f"{VERIFY_SECONDS}s.", "human_action": result.get("human_action")}
    elif outcome == "no_action_needed":
        say("   it had already recovered")
        publish(incident.id, "fixed", text="It had already recovered; no action was needed.", fix_actions=[])
        return {**record, "outcome": "no_action_needed"}
    text = result.get("reason") or record.get("error") or "The fixer could not restore the pipeline."
    say(f"{_BOLD}ESCALATED:{_RESET} {text}")
    publish(incident.id, "escalated", text=text, human_action=result.get("human_action"),
            fix_actions=record["actions"])
    return {**record, "outcome": "escalated"}


async def handle(queue: "asyncio.Queue[Incident]", args: argparse.Namespace, publish: IncidentPublisher,
                 watcher: Watcher) -> None:
    investigate = None
    if not args.no_investigate:
        from msight_agent.sentinel.investigator import investigate  # needs claude_agent_sdk
    fix_history: dict[str, list[float]] = defaultdict(list)
    while True:
        incident = await queue.get()
        say(f"{_BOLD}{_RED}INCIDENT {incident.id} [{incident.kind}]{_RESET} {incident.summary}")
        for entry in incident.timeline:
            say(f"   {entry}")
        say(f"   detected {incident.detected_at - incident.first_event_at:.0f}s after the first sign "
            f"(first sign {clock(incident.first_event_at)})")

        investigation = None
        if investigate is not None:
            say(f"{_BOLD}Claude ({args.model}) is investigating...{_RESET}")
            publish(incident.id, "investigating", model=args.model)
            investigation = await investigate(incident, say, model=args.model, mcp_url=args.mcp_url,
                                              max_turns=args.max_turns, max_budget_usd=args.max_budget,
                                              timeout_s=args.timeout,
                                              progress=lambda text: publish(incident.id, "progress", text=text))
            report = investigation.get("report")
            if report:
                say(f"{_BOLD}ROOT CAUSE ({report['category']}, {report['confidence']}):{_RESET} {report['root_cause']}")
                say(f"{_BOLD}SUGGESTED FIX (not applied):{_RESET} {report['suggested_fix']}")
            else:
                say(f"{_RED}No structured report: {investigation.get('error') or investigation.get('stop')}{_RESET}")
            for warning in investigation["checks"]:
                say(f"{_RED}CHECK:{_RESET} {warning}")
            say(f"   {len(investigation['tool_calls'])} tool calls, {investigation.get('duration_s', 0):.0f}s, "
                f"${investigation.get('cost_usd') or 0:.3f}")

        path = write_incident(incident.to_dict(), investigation)
        say(f"   report: {path}")
        if investigation is not None:
            publish(incident.id, "diagnosed", report=investigation.get("report"), checks=investigation["checks"],
                    error=investigation.get("error"), report_path=str(path), autonomy=args.autonomy,
                    cost_usd=investigation.get("cost_usd"), duration_s=round(investigation.get("duration_s", 0)))
            if args.autonomy == "fix":
                fix = await autofix(incident, investigation.get("report"), investigation["checks"], args,
                                    publish, watcher, fix_history)
                append_event(incident.id, "autofix", outcome=fix.get("outcome"), actions=fix.get("actions", []),
                             cost_usd=fix.get("cost_usd"), denied=fix.get("denied", []))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-investigate", action="store_true", help="detect and log only; don't call Claude")
    parser.add_argument("--autonomy", choices=("observe", "suggest", "fix"), default="suggest",
                        help="after a diagnosis: report only, offer a Fix-it button (default), or fix automatically")
    parser.add_argument("--model", default=None, help="Claude model for investigations")
    parser.add_argument("--sweep-seconds", type=float, default=30, help="pipeline health sweep interval")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-budget", type=float, default=1.0, help="USD cap per investigation")
    parser.add_argument("--timeout", type=float, default=300, help="seconds per investigation")
    parser.add_argument("--max-fixes", type=int, default=MAX_FIXES_PER_WINDOW,
                        help="automatic fixes per node per 30 min before it counts as flapping and goes to a "
                             "person (raise it for test suites that break the same node on purpose)")
    parser.add_argument("--api-url", default="http://localhost:8003", help="msight_api")
    parser.add_argument("--mcp-url", default="http://localhost:8000/sse", help="mcp_server (SSE)")
    args = parser.parse_args()
    if not args.no_investigate and args.model is None:
        from msight_agent.sentinel.investigator import DEFAULT_MODEL
        args.model = DEFAULT_MODEL

    broker = redis_kwargs()  # the MSight broker settings come from .env: read them before the scrub below
    # Importing msight_agent loaded all of .env into the environment, which the Claude
    # subprocess would inherit; keep only what the investigator needs.
    for name in secret_env_names(os.environ, dotenv_values(REPO_ROOT / ".env").keys()):
        os.environ.pop(name)

    queue: asyncio.Queue[Incident] = asyncio.Queue()
    publish = IncidentPublisher(broker, say)
    watcher = Watcher(queue, say, publish=publish, redis_settings=broker,
                      api_url=args.api_url, sweep_seconds=args.sweep_seconds)
    after = {"observe": "report only", "suggest": "offer Fix-it to a person",
             "fix": "FIX AUTOMATICALLY (scoped, verified)"}[args.autonomy]
    mode = "detect only" if args.no_investigate else f"investigate with {args.model}; {after}"
    say(f"{_BOLD}MSight sentinel{_RESET} watching containers, heartbeats, host-process nodes, pipeline health "
        f"(every {args.sweep_seconds:.0f}s), camera pictures, frame rates and agent services -- {mode}. "
        f"Reports: {INCIDENT_DIR}")
    # A deliberate stop (Ctrl+C, or SIGTERM from `systemctl stop`) tells the chat the monitor
    # was stopped on purpose; a crash leaves the marker so the chat reports it as down.
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    try:
        await asyncio.gather(watcher.run(), handle(queue, args, publish, watcher))
    except asyncio.CancelledError:
        watcher.mark_stopped()
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
