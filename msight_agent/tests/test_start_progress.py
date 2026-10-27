"""Starting the pipeline must stream progress lines (they reach the chat as SSE "log"
events); a start with no output looked frozen while images were pulled."""
import asyncio
from pathlib import Path

from msight_agent.msight.runtime import control_plane as cp_mod
from msight_agent.msight.runtime.control_plane import MSightControlPlane


class _FakeExecutor:
    async def start(self, name, cmd, cwd, env, needs_gpu=False, mounts=None, progress=None):
        if progress:
            await progress(f"docker output for {name}")
        return object()

    async def is_alive(self, handle):
        return True


def test_recompose_streams_progress(monkeypatch):
    monkeypatch.setattr(cp_mod, "_resolve_msight_path", lambda: Path("/msight"))
    cp = MSightControlPlane()
    monkeypatch.setattr(cp, "_executor_for", lambda invocation: _FakeExecutor())

    async def no_redis():
        return None
    monkeypatch.setattr(cp, "_ensure_redis", no_redis)

    lines: list[str] = []

    async def progress(text):
        lines.append(text)

    desired = [{"node_type": "detection_viewer", "name": "viewer",
                "config": {"subscribe_topic": "detection/x", "port": 9010}}]
    asyncio.run(cp.recompose(desired, progress=progress))
    assert lines == ["Starting viewer (detection_viewer)...", "docker output for viewer", "viewer started."]

    lines.clear()
    asyncio.run(cp.recompose(desired, progress=progress))  # already live: reused, still reported
    assert lines == ["viewer is already running."]
