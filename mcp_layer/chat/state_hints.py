"""The SESSION_STATE hint injected before each user message, so the model never
infers workflow progress from memory."""
import logging

from mcp_layer.chat.prompts import STATE_HINTS
from mcp_layer.mcptools.msight.calibration import calibration_label
from mcp_layer.mcptools.msight.control_plane import get_control_plane
from mcp_layer.mcptools.msight.paths import get_msight_path
from mcp_layer.mcptools.msight.pipeline import FIXED_PIPELINE_NODES
from mcp_layer.mcptools.msight.record_archive import DUMPER_NODE, PUSHER_NODE
from mcp_layer.validate_workflow_state import (
    AutoLabelingPhase, LabelingBackend, LabelingPath, WorkflowState, WORKFLOW_SPECS,
)

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


async def _msight_hint(state: WorkflowState) -> str:
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


def _auto_labeling_parts(state: WorkflowState) -> list[str]:
    al = state.auto_labeling
    if not al:
        return []
    parts = []
    if al.labeling_backend == LabelingBackend.BOTH:
        parts.append(STATE_HINTS["backend_both"])
    elif al.labeling_backend and not al.labeling_path:
        parts.append(STATE_HINTS["backend_confirmed_no_path"].format(backend=al.labeling_backend))
    elif al.labeling_backend:
        parts.append(f"backend={al.labeling_backend}")

    if al.labeling_path == LabelingPath.MANUAL and not al.manual_classes:
        export_fn = ("export_to_label_studio" if al.labeling_backend == LabelingBackend.LABEL_STUDIO
                     else "export_to_cvat")
        parts.append(STATE_HINTS["manual_no_classes"].format(
            export_fn=export_fn, dataset_name=state.dataset_name or "?"))
    elif al.labeling_path:
        parts.append(f"labeling_path={al.labeling_path}")

    if al.manual_classes and not al.cvat_task_id and not al.ls_task_ids and not al.phase:
        key = "manual_classes_export_confirmed" if al.export_confirmed else "manual_classes_awaiting_confirm"
        parts.append(STATE_HINTS[key].format(classes=", ".join(al.manual_classes)))

    if al.phase in (AutoLabelingPhase.ANNOTATING, AutoLabelingPhase.TRAINING):
        # COMPLETE is handled by the labels_imported check below.
        action = ("export complete — awaiting annotation" if al.phase == AutoLabelingPhase.ANNOTATING
                  else "auto-labeling complete — awaiting import")
        parts.append(STATE_HINTS["phase_locked"].format(
            phase=al.phase, action=action, workflow_name=state.workflow_name))

    if al.models_listed and not al.model_configured:
        parts.append(STATE_HINTS["models_listed_not_configured"])
    if al.model_configured:
        if al.auto_labeling_complete:
            parts.append("model=configured")
        elif al.run_awaiting_confirmation and not al.run_confirmed:
            parts.append(STATE_HINTS["model_configured_awaiting_run_confirm"])
        else:
            parts.append(STATE_HINTS["model_configured_ready"])
    if al.auto_labeling_complete:
        parts.append("auto_labeling=complete")
    if al.labeling_path == LabelingPath.AUTO:
        parts.append(f"localization={'enabled' if al.localization_enabled else 'not configured'}")
    if al.labels_imported:
        parts.append(STATE_HINTS["workflow_complete"].format(labeled_dataset_name=state.labeled_dataset_name))
    if not al.phase and not al.auto_labeling_complete and (al.labeling_path or al.model_configured):
        parts.append(STATE_HINTS["reconfigurable_zone_a"].format(workflow_name=state.workflow_name))
    return parts


async def build_state_hint(state: WorkflowState | None) -> str:
    """The SESSION_STATE line for this turn ("" if no workflow is selected)."""
    try:
        if state is None:
            state = WorkflowState.load()
        if not state.workflow_name:
            return ""
        parts = [f"workflow={state.workflow_name}"]
        spec = WORKFLOW_SPECS.get(state.workflow_name)
        if spec and not spec.requires_dataset:
            if state.workflow_name == "msight_pipeline":
                parts.append(await _msight_hint(state))
        else:
            parts.append(f"dataset={state.dataset_name}" if state.dataset_confirmed and state.dataset_name
                         else STATE_HINTS["dataset_not_confirmed"])
            parts += _auto_labeling_parts(state)
        return "SESSION_STATE: " + " | ".join(parts)
    except Exception as e:
        logging.warning(f"[STATE] Could not build state hint: {e}")
        return ""
