"""The SESSION_STATE hint injected before each user message, so the model never
infers workflow progress from memory. Each workflow builds its own part."""
import logging

from msight_agent.agent.prompts import STATE_HINTS
from msight_agent.agent.state import WorkflowState, WORKFLOW_SPECS
from msight_agent.auto_labeling.hints import auto_labeling_hint_parts
from msight_agent.msight.hints import msight_hint


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
                parts.append(await msight_hint(state))
        else:
            parts.append(f"dataset={state.dataset_name}" if state.dataset_confirmed and state.dataset_name
                         else STATE_HINTS["dataset_not_confirmed"])
            parts += auto_labeling_hint_parts(state)
        return "SESSION_STATE: " + " | ".join(parts)
    except Exception as e:
        logging.warning(f"[STATE] Could not build state hint: {e}")
        return ""
