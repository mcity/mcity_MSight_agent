"""Break a running MSight pipeline on purpose, for sentinel demos.

    python -m msight_agent.sentinel.inject kill rfdetr_detector     # admin kill (docker kill)
    python -m msight_agent.sentinel.inject exit rfdetr_detector     # process quits on its own (exit 0)
    python -m msight_agent.sentinel.inject oom rfdetr_detector      # memory limit below its usage
    python -m msight_agent.sentinel.inject freeze rfdetr_detector   # docker pause: "running" but frozen
    python -m msight_agent.sentinel.inject unfreeze rfdetr_detector
    python -m msight_agent.sentinel.inject kill-process <node>      # SIGKILL a host-process node
    python -m msight_agent.sentinel.inject restore                  # put the demo pipeline back

Stdlib only, so it runs from either venv.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import urllib.request

from msight_agent.shared.paths import NODE_LOG_DIR

API_URL = "http://localhost:8003"
OOM_LIMIT = "700m"   # below RF-DETR's working set, so it is killed as soon as it runs


class InjectError(RuntimeError):
    pass


def _docker(*args: str, check: bool = True) -> str:
    result = subprocess.run(["docker", *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        raise InjectError(f"docker {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _container(node: str) -> str:
    return node if node.startswith("msight-") else f"msight-{node}"


def kill(node: str) -> None:
    _docker("kill", _container(node))
    print(f"Sent SIGKILL to {_container(node)} through Docker (as an admin would).")


def exit_cleanly(node: str) -> None:
    # SIGINT from inside the container (no Docker kill event): msight_core stops its
    # listener and the process exits 0, which on-failure:3 doesn't restart.
    _docker("exec", _container(node), "kill", "-INT", "1")
    print(f"Made {_container(node)}'s process quit on its own (exit 0); Docker won't restart it.")


def oom(node: str, limit: str) -> None:
    # Applies to the running container and survives Docker's restarts of it, so each
    # restart is OOM-killed again until Docker's on-failure:3 gives up.
    _docker("update", "--memory", limit, "--memory-swap", limit, _container(node))
    print(f"Capped {_container(node)} at {limit} of memory; the kernel will OOM-kill it.")


def freeze(node: str, frozen: bool = True) -> None:
    _docker("pause" if frozen else "unpause", _container(node))
    print(f"{'Froze' if frozen else 'Unfroze'} {_container(node)}.")


def kill_process(node: str) -> None:
    pidfile = NODE_LOG_DIR / f"{node}.pid"
    if not pidfile.is_file():
        raise InjectError(f"No pidfile for '{node}' in {NODE_LOG_DIR}: it isn't a running host-process node.")
    pid = int(pidfile.read_text().strip())
    os.kill(pid, signal.SIGKILL)
    print(f"Sent SIGKILL to host-process node '{node}' (pid {pid}).")


def restore(video: str) -> None:
    names = _docker("ps", "-a", "--format", "{{.Names}}", "--filter", "name=msight-").split()
    for name in names:
        if name == "msight-redis":
            continue
        info = json.loads(_docker("inspect", name))[0]
        if info["State"].get("Paused"):
            _docker("unpause", name)
            print(f"Unpaused {name}.")
        # A memory cap outlives restarts, so a capped container is replaced, not kept.
        if (info["HostConfig"].get("Memory") or 0) > 0:
            _docker("rm", "-f", name)
            print(f"Removed {name} (it had a memory cap).")
    body = json.dumps({"video_input": video}).encode()
    request = urllib.request.Request(f"{API_URL}/pipeline/start", data=body,
                                     headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as resp:
        print(json.loads(resp.read()).get("message"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="fault", required=True)
    for name in ("kill", "exit", "freeze", "unfreeze", "kill-process"):
        sub.add_parser(name).add_argument("node")
    oom_parser = sub.add_parser("oom")
    oom_parser.add_argument("node")
    oom_parser.add_argument("--limit", default=OOM_LIMIT, help="memory cap; must be below the node's usage")
    restore_parser = sub.add_parser("restore")
    restore_parser.add_argument("--video", default=os.environ.get("MSIGHT_DEMO_VIDEO_PATH"),
                                help="demo video (default: MSIGHT_DEMO_VIDEO_PATH from .env)")
    args = parser.parse_args()
    try:
        _run(args)
    except InjectError as e:
        sys.exit(str(e))


def _run(args: argparse.Namespace) -> None:
    if args.fault == "kill":
        kill(args.node)
    elif args.fault == "exit":
        exit_cleanly(args.node)
    elif args.fault == "oom":
        oom(args.node, args.limit)
    elif args.fault in ("freeze", "unfreeze"):
        freeze(args.node, args.fault == "freeze")
    elif args.fault == "kill-process":
        kill_process(args.node)
    elif args.fault == "restore":
        if not args.video:
            sys.exit("No demo video: pass --video or set MSIGHT_DEMO_VIDEO_PATH in .env.")
        restore(args.video)


if __name__ == "__main__":
    main()
