"""Auto-labeling's part of the per-turn SESSION_STATE hint."""
from msight_agent.agent.prompts import STATE_HINTS
from msight_agent.agent.state import AutoLabelingPhase, LabelingBackend, LabelingPath, WorkflowState


def auto_labeling_hint_parts(state: WorkflowState) -> list[str]:
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

    if al.manual_classes and not al.cvat_task_ids and not al.ls_task_ids and not al.phase:
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
        if (al.labeling_path == LabelingPath.AUTO and not al.phase
                and not (al.cvat_task_ids or al.ls_task_ids)):
            export_fn = ("export_to_label_studio" if al.labeling_backend == LabelingBackend.LABEL_STUDIO
                         else "export_to_cvat")
            parts.append(STATE_HINTS["predictions_not_exported"].format(
                export_fn=export_fn, dataset_name=state.dataset_name or "?"))
    if al.labeling_path == LabelingPath.AUTO:
        parts.append(f"localization={'enabled' if al.localization_enabled else 'not configured'}")
    if al.labels_imported:
        parts.append(STATE_HINTS["workflow_complete"].format(labeled_dataset_name=state.labeled_dataset_name))
    if not al.phase and not al.auto_labeling_complete and (al.labeling_path or al.model_configured):
        parts.append(STATE_HINTS["reconfigurable_zone_a"].format(workflow_name=state.workflow_name))
    # Without this the model treats a failed run as a dead end and offers only a restart.
    if lr := state.failed_run():
        parts.append(STATE_HINTS["last_run_failed"].format(
            attempts=lr.attempts, error=lr.error,
            log_note=f" Logs: {lr.log_path}." if lr.log_path else ""))
    return parts
