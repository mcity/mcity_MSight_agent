"""Shared routing/sentinel primitives used by ChatPipeline and its per-workflow
handler mixins (pipeline_handlers/). Kept as a separate leaf module so those
mixins can import Sentinels/HardStop/etc. without a circular import back to
chat_pipeline.py, which imports the mixins.
"""
import re
from dataclasses import dataclass as _dc
from pathlib import Path
from typing import Optional

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.py"
MAIN_PATH   = Path(__file__).resolve().parents[1] / "main.py"

# Human-readable status emitted to the UI before each tool call.
TOOL_STATUS_MESSAGES: dict[str, str] = {
    "select_workflow":                       "Setting up workflow...",
    "switch_workflow":                       "Switching workflow...",
    "set_selected_dataset":                  "Confirming dataset...",
    "list_datasets":                         "Fetching available datasets...",
    "list_model_sources_and_models":         "Fetching available models...",
    "configure_auto_labeling":               "Configuring model selection...",
    "set_auto_labeling_hyperparams":         "Updating hyperparameters...",
    "set_msight_localization_config":        "Configuring MSight localization...",
    "run_auto_labeling":                     "Starting auto-labeling — this may take several minutes...",
    "export_to_cvat":                        "Exporting dataset to CVAT...",
    "import_from_cvat":                      "Importing annotations from CVAT...",
    "export_to_label_studio":                "Exporting dataset to Label Studio...",
    "import_from_label_studio":              "Importing annotations from Label Studio...",
    "get_labeling_backend":                  "Detecting annotation backend...",
    "set_labeling_backend":                  "Configuring annotation backend...",
    "set_labeling_path":                     "Setting labeling path...",
    "confirm_export":                         "Recording export consent...",
    "confirm_run":                            "Recording run consent...",
    "reset_workflow_state":                   "Resetting workflow state...",
    "launch_voxel51_session":                "Launching Voxel51 visualization...",
}

# Strip ANSI escape codes and bare CR from subprocess output.
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\r')

# Live-ops tools that actually change something -- used by chat_pipeline.py to
# force send_reply's source tag off when one succeeds this turn (confirming an
# action is operational, not informational), and by chat_server.py to tell
# real progress apart from "nothing happened" when the agentic loop's
# iteration budget runs out before a reply is produced.
WRITE_TOOLS: frozenset[str] = frozenset({
    "start_msight_pipeline", "stop_msight_pipeline",
    "add_msight_node", "remove_msight_node",
    "start_msight_recording", "stop_msight_recording",
    "start_msight_archiving", "stop_msight_archiving",
})


def unwrap_tool_output(raw) -> str:
    """Normalize any LLM/MCP output type to a plain string."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if hasattr(raw, "text"):
        return (raw.text or "").replace("\\n", "\n").strip()
    if isinstance(raw, list):
        parts = [unwrap_tool_output(x) for x in raw]
        return "\n".join(p for p in parts if p).strip()
    if isinstance(raw, dict):
        if "text" in raw and isinstance(raw["text"], str):
            return raw["text"].replace("\\n", "\n").strip()
        if "content" in raw and isinstance(raw["content"], list):
            return unwrap_tool_output(raw["content"])
        if "data" in raw and isinstance(raw["data"], dict) and "msg" in raw["data"]:
            return str(raw["data"]["msg"]).replace("\\n", "\n").strip()
        for key in ("message", "detail"):
            if key in raw and isinstance(raw[key], str):
                return raw[key].replace("\\n", "\n").strip()
    return str(raw).strip()


# Each _handle_* returns (raw_str, list[ToolRouting]). _orchestrate does two
# passes: first all Injections (context), then last HardStop wins (so an explicit
# set_labeling_backend can override auto-detect when batched with set_selected_dataset).


@_dc
class Injection:
    """Inject a system context message; the agentic loop continues."""
    message: str


@_dc
class HardStop:
    """Return this reply directly to the user; the agentic loop ends."""
    reply: str


@_dc
class FallThrough:
    """No pipeline reply; let the final LLM pass summarize the tool result."""
    pass


ToolRouting = Injection | HardStop | FallThrough


@_dc
class ConfirmGateSpec:
    """Describes one consent-gate's bookkeeping shape -- captures how
    msight_pipeline's and auto_labeling's *already nearly-identical*
    run-confirmation patterns diverge, as data, instead of two independently
    hand-copied implementations. See ChatPipeline._check_confirm_gate /
    _clear_confirm_gate (chat_pipeline.py) for the generic mechanism this
    describes.

    Not yet wired into either existing call site -- pipeline_handlers/
    msight_pipeline.py and pipeline_handlers/auto_labeling.py keep their
    current, already-verified inline logic. This exists so a *future* third
    consent gate (e.g. around a destructive generic node operation, if that's
    ever needed) doesn't grow a third hand-copied 40-line implementation.
    """
    state_path: str             # attribute name on WorkflowState, e.g. "msight_pipeline"
    confirmed_attr: str         # e.g. "run_confirmed"
    awaiting_attr: str          # e.g. "run_awaiting_confirmation"
    requested_at_attr: Optional[str] = None  # per-field TTL timestamp attr, or None = defer to the global config.py-mtime TTL
    ttl_seconds: int = 300


# Documents the two existing patterns' actual shape (see
# pipeline_handlers/msight_pipeline.py:_handle_start_msight_pipeline and
# pipeline_handlers/auto_labeling.py's run-confirmation handlers) --
# msight_pipeline has a dedicated 300s TTL field; auto_labeling has none and
# relies on WorkflowState.load()'s separate global 3600s config.py-mtime TTL
# block instead. Reference only until a gate actually migrates onto this.
CONFIRM_GATES: dict[str, ConfirmGateSpec] = {
    "msight_pipeline_run": ConfirmGateSpec(
        state_path="msight_pipeline", confirmed_attr="run_confirmed",
        awaiting_attr="run_awaiting_confirmation",
        requested_at_attr="run_confirmation_requested_at", ttl_seconds=300,
    ),
    "auto_labeling_run": ConfirmGateSpec(
        state_path="auto_labeling", confirmed_attr="run_confirmed",
        awaiting_attr="run_awaiting_confirmation",
    ),
}


class Sentinels:
    """Prefix strings returned by MCP tools to signal specific error/state conditions."""
    DATASET_NOT_FOUND         = "DATASET_NOT_FOUND"
    EXPORT_NEEDS_CONFIRMATION = "EXPORT_NEEDS_CONFIRMATION"
    RUN_NEEDS_CONFIRMATION    = "RUN_NEEDS_CONFIRMATION"
    LS_BACKEND_ERROR          = "LS_BACKEND_ERROR"
    LS_AUTH_ERROR             = "LS_AUTH_ERROR"
    LS_CONNECTION_ERROR       = "LS_CONNECTION_ERROR"
    CVAT_TASK_LIMIT_REACHED   = "CVAT_TASK_LIMIT_REACHED"
    CVAT_STORAGE_LIMIT_REACHED = "CVAT_STORAGE_LIMIT_REACHED"
    CVAT_FORBIDDEN            = "CVAT_FORBIDDEN"
    CVAT_AUTH_ERROR           = "CVAT_AUTH_ERROR"
    CVAT_NOT_FOUND            = "CVAT_NOT_FOUND"
    CVAT_CONNECTION_ERROR     = "CVAT_CONNECTION_ERROR"
    CVAT_TIMEOUT_ERROR        = "CVAT_TIMEOUT_ERROR"
    BACKEND_NOT_SET           = "BACKEND_NOT_SET"
    LS_NO_ANNOTATIONS         = "LS_NO_ANNOTATIONS"
    MSIGHT_LOCALIZATION_ERROR = "MSIGHT_LOCALIZATION_ERROR"
