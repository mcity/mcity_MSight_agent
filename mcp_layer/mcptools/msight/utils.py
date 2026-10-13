"""Small helpers shared across the msight package."""
import asyncio
import os
from typing import Optional


async def run_cmd(*argv: str, timeout: Optional[float] = None, merge_stderr: bool = True,
                  cwd: Optional[str] = None) -> tuple[Optional[int], str]:
    """Run a command without a shell. Returns (exit code, output); the code is
    None on timeout and 127 if the program isn't installed."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=cwd, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT if merge_stderr else asyncio.subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return 127, f"'{argv[0]}' is not installed on this host."
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return None, f"timed out after {timeout:g}s"
    return proc.returncode, out.decode(errors="replace")


def redis_kwargs() -> dict:
    """Connection settings for the MSight message broker, from the environment."""
    return {
        "host": os.environ.get("MSIGHT_REDIS_MESSAGE_BROKER_HOST", "localhost"),
        "port": int(os.environ.get("MSIGHT_REDIS_MESSAGE_BROKER_PORT", 6379)),
        "db": int(os.environ.get("MSIGHT_REDIS_MESSAGE_BROKER_DB", 0)),
    }


def topic_list(value) -> list[str]:
    """msight_core registers publish/subscribe topics as a string, a list, or null."""
    if not value:
        return []
    return [value] if isinstance(value, str) else [t for t in value if t]


def ago(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds}s ago" if seconds < 120 else f"{seconds // 60} min ago"
