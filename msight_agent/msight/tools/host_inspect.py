"""Read-only host inspection for the long tail the coded checks don't cover
(drivers, disk, ports, Docker, services) -- modeled on HolmesGPT's bash toolset.

Commands run without a shell, only if the program AND its arguments match the
allowlist below; nothing here can change host state or read arbitrary files.
"""
import re
import shlex
from typing import Callable, Optional

from msight_agent.shared.mcp import mcp
from msight_agent.shared.mcp import error_json, ok_json
from msight_agent.msight.runtime.utils import run_cmd

_TIMEOUT_SECONDS = 10
_MAX_OUTPUT_CHARS = 8000
_SHELL_CHARS = set("|;&<>`$\\\n")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+*-]*$")  # units, containers, package patterns
_REDACT_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL)[A-Z0-9_]*)=(\S+)"
)
_AWS_KEY_RE = re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b")


def _flags_only(allowed: set[str]) -> Callable[[list[str]], Optional[str]]:
    def check(args: list[str]) -> Optional[str]:
        bad = [a for a in args if a not in allowed]
        return f"only these flags are allowed: {sorted(allowed)}" if bad else None
    return check


def _names(args: list[str]) -> Optional[str]:
    bad = [a for a in args if not _NAME_RE.match(a)]
    return f"invalid name(s): {bad}" if bad else None


def _dpkg(args: list[str]) -> Optional[str]:
    if not args or args[0] != "-l":
        return "only 'dpkg -l [package-pattern ...]' is allowed"
    return _names(args[1:])


def _nvidia_smi(args: list[str]) -> Optional[str]:
    # Many nvidia-smi flags change GPU settings (-pm, -r, -c, ...); allow queries only.
    for a in args:
        if a not in ("-L", "-q") and not a.startswith(("--query-gpu=", "--format=")):
            return "only nvidia-smi, -L, -q, --query-gpu=..., --format=... are allowed"
    return None


def _df(args: list[str]) -> Optional[str]:
    for a in args:
        if a.startswith("-"):
            if a not in ("-h", "-i", "-T"):
                return "only -h, -i, -T flags are allowed"
        elif not a.startswith("/") or ".." in a:
            return "df paths must be absolute"
    return None


def _ss(args: list[str]) -> Optional[str]:
    if all(re.fullmatch(r"-[ltunpa]+", a) for a in args):
        return None
    return "only combined -l/-t/-u/-n/-p/-a flags are allowed (e.g. -ltnp)"


_DOCKER_SUBCOMMANDS = {"ps", "inspect", "logs", "images", "info", "version", "stats"}


def _docker(args: list[str]) -> Optional[str]:
    if not args or args[0] not in _DOCKER_SUBCOMMANDS:
        return f"only docker {sorted(_DOCKER_SUBCOMMANDS)} are allowed"
    sub, rest = args[0], args[1:]
    if sub == "stats" and "--no-stream" not in rest:
        return "docker stats needs --no-stream"
    i = 0
    while i < len(rest):
        a = rest[i]
        if a in ("--tail", "-n", "--since", "-f", "--format"):
            if a == "-f" and sub == "logs":
                return "docker logs -f (follow) is not allowed"
            i += 2  # flag value: a count, duration or Go template -- passed as one argv item
            continue
        if a.startswith("-"):
            if a not in ("-a", "--all", "-t", "--timestamps", "--no-stream", "--no-trunc", "-q"):
                return f"docker flag '{a}' is not allowed"
        elif _names([a]):
            return f"invalid container/image name '{a}'"
        i += 1
    return None


def _systemctl(args: list[str]) -> Optional[str]:
    if len(args) < 2 or args[0] not in ("status", "is-active", "is-enabled"):
        return "only 'systemctl status|is-active|is-enabled <unit>' is allowed"
    return _names([a for a in args[1:] if a != "--no-pager"])


def _journalctl(args: list[str]) -> Optional[str]:
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-u", "-n", "-p", "--since"):
            if i + 1 >= len(args) or not re.fullmatch(r"[A-Za-z0-9_.:@ -]+", args[i + 1]):
                return f"'{a}' needs a simple value"
            i += 2
            continue
        if a not in ("-k", "-b", "--no-pager"):
            return "only -u, -n, -p, --since, -k, -b are allowed"
        i += 1
    if "-u" not in args and "-k" not in args:
        return "journalctl needs -u <unit> or -k (whole-system journal is too broad)"
    return None


_ALLOWLIST: dict[str, Callable[[list[str]], Optional[str]]] = {
    "uname": _flags_only({"-a", "-r", "-s", "-v", "-m"}),
    "uptime": _flags_only(set()),
    "nproc": _flags_only(set()),
    "lsmod": _flags_only(set()),
    "lspci": _flags_only({"-k", "-nn", "-v"}),
    "modinfo": lambda a: _names([x for x in a if x not in ("-F", "version", "vermagic")]),
    "mokutil": _flags_only({"--sb-state"}),
    "dpkg": _dpkg,
    "nvidia-smi": _nvidia_smi,
    "df": _df,
    "free": _flags_only({"-h", "-m"}),
    "ss": _ss,
    "docker": _docker,
    "systemctl": _systemctl,
    "journalctl": _journalctl,
}


def validate(command: str) -> tuple[Optional[list[str]], Optional[str]]:
    """(argv, None) if allowed, else (None, reason)."""
    if any(c in _SHELL_CHARS for c in command):
        return None, "shell operators (| ; & < > ` $ \\) are not allowed -- run one plain command"
    try:
        argv = shlex.split(command)
    except ValueError as e:
        return None, f"could not parse command: {e}"
    if not argv:
        return None, "empty command"
    program, args = argv[0], argv[1:]
    if program not in _ALLOWLIST:
        return None, f"'{program}' is not allowed. Allowed programs: {sorted(_ALLOWLIST)}"
    reason = _ALLOWLIST[program](args)
    if reason:
        return None, f"{program}: {reason}"
    if program in ("systemctl", "journalctl") and "--no-pager" not in argv:
        argv.append("--no-pager")
    return argv, None


def redact(text: str) -> str:
    return _AWS_KEY_RE.sub("<redacted>", _REDACT_RE.sub(r"\1=<redacted>", text))


async def run_inspection(command: str) -> dict:
    argv, reason = validate(command)
    if reason:
        return {"allowed": False, "reason": reason}
    code, out = await run_cmd(*argv, timeout=_TIMEOUT_SECONDS)
    text = redact(out)
    return {"allowed": True, "command": shlex.join(argv), "exit_code": code,
            "output": text[-_MAX_OUTPUT_CHARS:], "truncated": len(text) > _MAX_OUTPUT_CHARS}


@mcp.tool()
async def inspect_host(command: str) -> str:
    result = await run_inspection(command)
    if not result["allowed"]:
        return error_json(f"Command not allowed: {result['reason']}")
    return ok_json(**{k: v for k, v in result.items() if k != "allowed"})
