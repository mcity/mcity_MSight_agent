"""System prompt assembly: the agent's base rules + the active workflow's prompt,
and the per-state hint templates from agent/prompts/state_hints.txt.

New workflow: add it to WORKFLOW_SPECS with prompt_file pointing at its own prompts/ folder.
"""
import os
import re
from pathlib import Path

from msight_agent.shared.host_utils import is_cloud_deployment
from msight_agent.agent.state import WORKFLOW_SPECS

_AGENT_PROMPTS = Path(__file__).resolve().parent / "prompts"
_PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _demo_video_hint() -> str:
    """Demo video is deployment-specific, so it comes from .env."""
    path = os.environ.get("MSIGHT_DEMO_VIDEO_PATH", "").strip()
    if path:
        return (
            f"The built-in sample video is the fixed default for Demo mode, "
            f"always: video_input='{path}'. Demo takes zero "
            f"input from the user, ever — explain what's about to run (this "
            f"default source, that it's the demo calibration) and call "
            f"start_msight_pipeline with it in the same reply. Do NOT ask the "
            f"user to confirm or choose first — Demo has no consent exchange "
            f"(see STEP 3); asking here just reintroduces the round trip that "
            f"removes."
        )
    return (
        "No default demo video is configured on this host "
        "(MSIGHT_DEMO_VIDEO_PATH is not set in .env) — Demo mode can't run "
        "right now. Tell the user plainly that Demo isn't available on this "
        "host (misconfiguration, not something they can fix), and offer "
        "\"run your own pipeline\" instead. Do not ask them for a source "
        "within Demo — Demo never takes one."
    )


def _environment_hint() -> str:
    """Local paths are only offered when the user can reach this host's filesystem."""
    if is_cloud_deployment():
        return ("This host is a cloud sandbox — the user has no access to its filesystem. "
                "Never offer or accept a local file/folder path as a video source here; "
                "only an rtsp:// URL or the 📤 upload button are valid ways for this user "
                "to specify their own source.")
    return ("This host is local/on-prem — the user may have direct filesystem access, "
            "so a local file/folder path is a valid video source option here.")


def _strip_marked_block(text: str, tag: str, keep: bool) -> str:
    """Strip <!-- TAG:START/END --> markers; keep=False also drops the content between them."""
    start, end = f"<!-- {tag}:START -->", f"<!-- {tag}:END -->"
    if keep:
        return text.replace(start + "\n", "").replace(end + "\n", "")
    return re.sub(re.escape(start) + r".*?" + re.escape(end) + r"\n?", "", text, flags=re.DOTALL)


def _msight_prompt_by_mode(text: str) -> dict[str, str]:
    """STEP2A is the Demo section, STEP2B "run your own pipeline"; a mode shows only its own."""
    text = text.replace("{DEMO_VIDEO_HINT}", _demo_video_hint()).replace("{ENVIRONMENT_HINT}", _environment_hint())

    def only(keep_a: bool, keep_b: bool) -> str:
        return _strip_marked_block(_strip_marked_block(text, "STEP2A", keep_a), "STEP2B", keep_b)

    return {"": only(True, True), "demo": only(True, False), "custom": only(False, True)}


def _load_state_hints() -> dict[str, str]:
    """Parse state_hints.txt ("=== name ===" sections) into {name: template}."""
    sections = re.split(r"^=== (\w+) ===\s*$", (_AGENT_PROMPTS / "state_hints.txt").read_text(), flags=re.MULTILINE)
    return {name: body.strip() for name, body in zip(sections[1::2], sections[2::2])}


_WORKFLOW_PROMPTS = {
    name: (_PACKAGE_DIR / spec.prompt_file).read_text()
    for name, spec in WORKFLOW_SPECS.items()
}
_MSIGHT_PROMPT_BY_MODE = _msight_prompt_by_mode(_WORKFLOW_PROMPTS["msight_pipeline"])

BASE_PROMPT = (_AGENT_PROMPTS / "base_prompt.txt").read_text().replace(
    "{WORKFLOW_LIST}",
    "\n".join(f"{i + 1}. {spec.label} (internal name: {name})"
              for i, (name, spec) in enumerate(WORKFLOW_SPECS.items())),
)
STATE_HINTS = _load_state_hints()


def build_system_prompt(state) -> str:
    """Base rules + the active workflow's prompt, if one is selected and registered."""
    if not state or state.workflow_name not in _WORKFLOW_PROMPTS:
        return BASE_PROMPT
    if state.workflow_name == "msight_pipeline":
        mode = state.msight_pipeline.mode if state.msight_pipeline else ""
        workflow_text = _MSIGHT_PROMPT_BY_MODE.get(mode, _MSIGHT_PROMPT_BY_MODE[""])
    else:
        workflow_text = _WORKFLOW_PROMPTS[state.workflow_name]
    return BASE_PROMPT + "\n\n" + workflow_text
