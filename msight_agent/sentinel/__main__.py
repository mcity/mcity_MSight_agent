"""python -m msight_agent.sentinel [--no-investigate] [--model ...]

Watches the deployment; each incident is announced to the chat feed as soon as it's
seen, investigated by Claude (one at a time, in order) with live progress, and
written to data/logs/incidents/."""
import argparse
import asyncio
import os
import sys
import time

from dotenv import dotenv_values

from msight_agent.sentinel.checks import secret_env_names
from msight_agent.sentinel.incident_log import INCIDENT_DIR, write_incident
from msight_agent.sentinel.stream import IncidentPublisher, redis_settings
from msight_agent.sentinel.watcher import Incident, Watcher, clock
from msight_agent.shared.paths import REPO_ROOT

_BOLD, _RED, _RESET = ("\033[1m", "\033[31m", "\033[0m") if sys.stdout.isatty() else ("", "", "")


def say(line: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {line}", flush=True)


async def handle(queue: "asyncio.Queue[Incident]", args: argparse.Namespace, publish: IncidentPublisher) -> None:
    investigate = None
    if not args.no_investigate:
        from msight_agent.sentinel.investigator import investigate  # needs claude_agent_sdk
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
                    error=investigation.get("error"), report_path=str(path),
                    cost_usd=investigation.get("cost_usd"), duration_s=round(investigation.get("duration_s", 0)))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-investigate", action="store_true", help="detect and log only; don't call Claude")
    parser.add_argument("--model", default=None, help="Claude model for investigations")
    parser.add_argument("--sweep-seconds", type=float, default=30, help="pipeline health sweep interval")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-budget", type=float, default=1.0, help="USD cap per investigation")
    parser.add_argument("--timeout", type=float, default=300, help="seconds per investigation")
    parser.add_argument("--api-url", default="http://localhost:8003", help="msight_api")
    parser.add_argument("--mcp-url", default="http://localhost:8000/sse", help="mcp_server (SSE)")
    args = parser.parse_args()
    if not args.no_investigate and args.model is None:
        from msight_agent.sentinel.investigator import DEFAULT_MODEL
        args.model = DEFAULT_MODEL

    redis_kwargs = redis_settings()  # from .env, so read before the scrub below
    # Importing msight_agent loaded all of .env into the environment, which the Claude
    # subprocess would inherit; keep only what the investigator needs.
    for name in secret_env_names(os.environ, dotenv_values(REPO_ROOT / ".env").keys()):
        os.environ.pop(name)

    queue: asyncio.Queue[Incident] = asyncio.Queue()
    publish = IncidentPublisher(redis_kwargs, say)
    watcher = Watcher(queue, say, publish=publish, redis_settings=redis_kwargs,
                      api_url=args.api_url, sweep_seconds=args.sweep_seconds)
    mode = "detect only" if args.no_investigate else f"investigate with {args.model}, read-only"
    say(f"{_BOLD}MSight sentinel{_RESET} watching containers, heartbeats, host-process nodes, pipeline health "
        f"(every {args.sweep_seconds:.0f}s) and agent services -- {mode}. Reports: {INCIDENT_DIR}")
    await asyncio.gather(watcher.run(), handle(queue, args, publish))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
