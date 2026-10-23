"""MSight's part of the per-turn SESSION_STATE hint: the checklist, live liveness
(the persisted *_active flags drift when things change outside the chat), and any
structural pipeline problems."""
import logging

from msight_agent.agent.prompts import STATE_HINTS
from msight_agent.agent.state import WorkflowState
from msight_agent.msight.runtime.calibration import calibration_label
from msight_agent.msight.runtime.control_plane import get_control_plane
from msight_agent.msight.runtime.paths import get_msight_path
from msight_agent.msight.tools.pipeline import FIXED_PIPELINE_NODES
from msight_agent.msight.tools.record_archive import DUMPER_NODE, PUSHER_NODE

_PENDING = "pending (will auto-start when pipeline starts)"


async def _msight_liveness() -> dict[str, bool]:
    """Live recording / archiving / pipeline state (one docker call). The persisted
    *_active flags drift when things change outside the chat (dashboard, crash)."""
    empty = {"recording": False, "archiving": False, "pipeline": False}
    if get_msight_path()[1]:
        return empty
    try:
        snap = await get_control_plane().snapshot([DUMPER_NODE, PUSHER_NODE, *FIXED_PIPELINE_NODES])
    except Exception:
        return empty
    return {
        "recording": snap[DUMPER_NODE]["alive"],
        "archiving": snap[PUSHER_NODE]["alive"],
        "pipeline": any(snap[n]["alive"] for n in FIXED_PIPELINE_NODES),
    }


async def _msight_problems_hint() -> str:
    """Structural graph problems (dead nodes, unpublished inputs)."""
    if get_msight_path()[1]:
        return ""
    try:
        problems = await get_control_plane().structural_check()
    except Exception:
        return ""
    if not problems:
        return ""
    return (
        " | PIPELINE_PROBLEMS: " + " ".join(p["evidence"] for p in problems)
        + " -- this is very likely the cause of any 'frozen'/'not working'/'no detections' "
        "symptom. Call diagnose_msight_pipeline for the ranked root cause before answering; "
        "don't report the nodes that ARE running as if everything's fine."
    )


def _source_label(mp) -> str:
    if mp.rtsp_url:
        return f"rtsp_url:{mp.rtsp_url}"
    return f"video_input:{mp.video_input}" if mp.video_input else ""


async def msight_hint(state: WorkflowState) -> str:
    mp = state.msight_pipeline
    if mp is None:
        return (f"msight_checklist(mode=not set, source=not set, {calibration_label(prefixed=True)}, "
                "recording=not active, archiving=not active)")

    live = await _msight_liveness()
    parts = [f"mode={mp.mode or 'not set'}", f"source={_source_label(mp) or 'not set'}",
             calibration_label(prefixed=True)]
    if mp.sensor_name:
        parts.append(f"sensor_name={mp.sensor_name}")
    parts.append(f"recording={'active' if live['recording'] else _PENDING if mp.recording_pending else 'not active'}")
    parts.append(f"archiving={'active' if live['archiving'] else _PENDING if mp.archiving_pending else 'not active'}")
    parts.append(f"pipeline_running={live['pipeline']}")

    # Persisted flags are a cache; resync any that drifted from live state.
    drifted = False
    for attr, value in (("recording_active", live["recording"]), ("archiving_active", live["archiving"]),
                        ("pipeline_running", live["pipeline"])):
        if getattr(mp, attr) != value:
            setattr(mp, attr, value)
            drifted = True
    if drifted:
        try:
            state.save()
        except Exception as e:
            logging.warning(f"[STATE] Could not save resynced msight flags: {e}")

    hint = "msight_checklist(" + ", ".join(parts) + ")"
    if mp.run_awaiting_confirmation and not mp.run_confirmed:
        hint += " | " + STATE_HINTS["msight_run_awaiting_confirm"]
    elif live["pipeline"]:
        hint += " | " + STATE_HINTS["msight_pipeline_running"].format(source=_source_label(mp))
    return hint + await _msight_problems_hint()
