"""
Pydantic state machine over WORKFLOW_STATE in config.py.
Provides typed schemas, precondition guards, tool input validation, and persistence.
"""

import ast
import importlib
import logging
from pathlib import Path
from typing import ClassVar, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

import config.config as _cc

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.py"


class LabelingBackend:
    """String constants for annotation backend names.

    Values match the Pydantic Literal fields in AutoLabelingState/SetLabelingBackendInput;
    those Literal definitions are the authoritative schema and are NOT changed here.
    """
    CVAT         = "cvat"
    LABEL_STUDIO = "label_studio"
    BOTH         = "both"   # sentinel: both backends available, user must choose
    NONE         = "none"   # sentinel: no credentials found


class AutoLabelingPhase:
    """String constants for auto-labeling workflow phase names.

    Values match the Pydantic Literal field in AutoLabelingState; that definition
    is the authoritative schema and is NOT changed here.
    """
    PENDING    = ""            # Zone A — mutable configuration
    ANNOTATING = "annotating"  # post-export, locked until import
    TRAINING   = "training"    # post-run, locked until import
    COMPLETE   = "complete"    # terminal state


class LabelingPath:
    """String constants for auto-labeling path names.

    Values match the Pydantic Literal field in AutoLabelingState; that definition
    is the authoritative schema and is NOT changed here.
    """
    MANUAL = "manual"
    AUTO   = "auto"


class SetAutoLabelingHyperparamsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Optional[list[Literal["train", "inference"]]] = None
    epochs: Optional[int] = Field(None, gt=0, le=1000)
    early_stop_patience: Optional[int] = Field(None, gt=0)
    early_stop_threshold: Optional[float] = Field(None, ge=0)
    learning_rate: Optional[float] = Field(None, gt=0)
    weight_decay: Optional[float] = Field(None, ge=0)
    max_grad_norm: Optional[float] = Field(None, gt=0)


class SetLabelingBackendInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Literal["cvat", "label_studio"]


class ConfigureAutoLabelingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selected_source: Literal[
        "ultralytics", "hf_models_objectdetection", "custom_codetr", "roboflow"
    ]
    selected_model: str = Field(min_length=1)


TOOL_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "set_auto_labeling_hyperparams": SetAutoLabelingHyperparamsInput,
    "configure_auto_labeling": ConfigureAutoLabelingInput,
    "set_labeling_backend": SetLabelingBackendInput,
}


def validate_tool_input(fn_name: str, fn_args: dict) -> tuple[bool, str, dict]:
    model_cls = TOOL_INPUT_MODELS.get(fn_name)
    if model_cls is None:
        return True, "", fn_args
    try:
        validated = model_cls.model_validate(fn_args)
        cleaned = {k: v for k, v in validated.model_dump().items() if v is not None}
        return True, "", cleaned
    except Exception as e:
        return False, f"Invalid arguments for {fn_name}: {e}", fn_args


class AutoLabelingState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    labeling_path: Literal["manual", "auto", ""] = ""
    labeling_backend: Literal["cvat", "label_studio", "both", ""] = ""
    manual_classes: list[str] = []
    models_listed: bool = False
    model_configured: bool = False
    hyperparams_confirmed: bool = False
    auto_labeling_complete: bool = False
    cvat_task_id: int = 0
    ls_task_ids: list[int] = []
    labels_imported: bool = False
    export_confirmed: bool = False
    run_confirmed: bool = False
    # True while the pre-run confirmation summary is shown; gates confirm_run tool.
    run_awaiting_confirmation: bool = False
    model_source: str = ""
    model_name: str = ""
    # "" = Zone A (mutable), "annotating" = post-export, "training" = post-run, "complete" = terminal
    phase: Literal["", "annotating", "training", "complete"] = ""

    _RESET_SEQUENCE: ClassVar[list[str]] = [
        "models_listed",
        "model_configured",
        "hyperparams_confirmed",
        "auto_labeling_complete",
        "labels_imported",
    ]

    def reset_from_step(self, step: str) -> None:
        """Clear all pipeline flags at and after `step` in the reset sequence."""
        try:
            idx = self._RESET_SEQUENCE.index(step)
        except ValueError:
            return
        for field in self._RESET_SEQUENCE[idx:]:
            setattr(self, field, False)

    def can_configure_auto_labeling(self) -> tuple[bool, str]:
        if not self.models_listed:
            return False, (
                "The available models must be listed before configuring. "
                "Please call list_model_sources_and_models first so the user "
                "can select from the actual available models."
            )
        return True, ""

    def can_set_auto_labeling_hyperparams(self) -> tuple[bool, str]:
        if not self.model_configured:
            return False, (
                "A model must be configured before setting hyperparameters. "
                "Please call configure_auto_labeling first to select a model."
            )
        return True, ""

    def can_run_auto_labeling(
        self, dataset_confirmed: bool, dataset_name: str
    ) -> tuple[bool, str]:
        if not dataset_confirmed or not dataset_name:
            return False, (
                f"No dataset has been confirmed for this session. "
                f"Config currently points to '{dataset_name or 'none'}'. "
                f"Please confirm the correct dataset name and call "
                f"set_selected_dataset first."
            )
        if not self.model_configured:
            return False, (
                "A model source and model must be configured before running "
                "auto-labeling. Please select a model first."
            )
        if self.labeling_path == LabelingPath.MANUAL:
            return False, (
                "Auto-labeling cannot run on the manual labeling path. "
                "Please export to the annotation tool and annotate manually."
            )
        return True, ""

    def can_export_to_cvat(self, with_predictions: bool) -> tuple[bool, str]:
        if with_predictions and not self.auto_labeling_complete:
            return False, (
                "Auto-labeling must complete before exporting predictions to CVAT. "
                "Please run auto-labeling first."
            )
        return True, ""

    def can_import_from_cvat(self) -> tuple[bool, str]:
        if self.cvat_task_id == 0:
            return False, (
                "The dataset must be exported to CVAT before importing annotations. "
                "Please export to CVAT first."
            )
        return True, ""

    def can_export_to_label_studio(self, with_predictions: bool) -> tuple[bool, str]:
        if with_predictions and not self.auto_labeling_complete:
            return False, (
                "Auto-labeling must complete before exporting predictions to Label Studio. "
                "Please run auto-labeling first."
            )
        return True, ""

    def can_import_from_label_studio(self) -> tuple[bool, str]:
        if not self.ls_task_ids:
            return False, (
                "The dataset must be exported to Label Studio before importing annotations. "
                "Please export to Label Studio first."
            )
        return True, ""


VALID_WORKFLOW = Literal["auto_labeling", ""]


class WorkflowState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_name: VALID_WORKFLOW = ""
    dataset_name: str = ""
    dataset_confirmed: bool = False
    labeled_dataset_name: str = ""

    auto_labeling: Optional[AutoLabelingState] = None

    # Triggers a WORKFLOW_RESET context injection in chat_server on next request.
    workflow_just_reset: bool = False

    @model_validator(mode="after")
    def dataset_confirmed_requires_name(self) -> "WorkflowState":
        if self.dataset_confirmed and not self.dataset_name:
            raise ValueError(
                "dataset_confirmed cannot be True when dataset_name is empty."
            )
        return self

    def can_confirm_dataset(self) -> tuple[bool, str]:
        if not self.workflow_name:
            return False, (
                "A workflow must be selected before confirming a dataset. "
                "Please select a workflow first."
            )
        return True, ""

    def reset_auto_labeling_from(self, step: str) -> None:
        if self.auto_labeling is not None:
            self.auto_labeling.reset_from_step(step)
        self.save()

    def valid_tool_names(self) -> set[str] | None:
        """Return valid tools for the current step, or None to expose all tools."""
        ALWAYS = {"send_reply", "switch_workflow"}

        if not self.workflow_name:
            return ALWAYS | {"select_workflow"}

        if not self.dataset_confirmed:
            return ALWAYS | {"set_selected_dataset", "list_datasets"}

        if self.workflow_name == "auto_labeling":
            return self._auto_labeling_tools(ALWAYS)

        return None

    def _auto_labeling_tools(self, ALWAYS: set[str]) -> set[str]:
        al = self.auto_labeling
        backend = (al.labeling_backend if al else "") or ""
        ls = backend == LabelingBackend.LABEL_STUDIO

        if al and al.phase == AutoLabelingPhase.COMPLETE:
            return ALWAYS | {"launch_voxel51_session"}
        if al and al.phase in (AutoLabelingPhase.ANNOTATING, AutoLabelingPhase.TRAINING):
            import_tool = "import_from_label_studio" if ls else "import_from_cvat"
            return ALWAYS | {import_tool}

        # list_datasets is excluded after dataset selection to prevent LLM listing loops.
        if not al or not al.labeling_path:
            if backend in (LabelingBackend.CVAT, LabelingBackend.LABEL_STUDIO):
                # Backend confirmed, awaiting path selection.
                return ALWAYS | {"set_selected_dataset", "set_labeling_path", "set_labeling_backend"}
            # Backend not yet confirmed — user must pick one first.
            return ALWAYS | {"set_selected_dataset", "get_labeling_backend", "set_labeling_backend"}

        if al.labeling_path == LabelingPath.MANUAL:
            if al.labels_imported:
                return ALWAYS | {"launch_voxel51_session"}
            if ls:
                if al.ls_task_ids:
                    return ALWAYS | {"import_from_label_studio"}
                if al.manual_classes and not al.export_confirmed:
                    return ALWAYS | {"confirm_export", "set_selected_dataset", "set_labeling_backend", "set_labeling_path"}
                return ALWAYS | {"set_selected_dataset", "export_to_label_studio", "set_labeling_backend", "set_labeling_path"}
            else:
                if al.cvat_task_id > 0:
                    return ALWAYS | {"import_from_cvat"}
                if al.manual_classes and not al.export_confirmed:
                    return ALWAYS | {"confirm_export", "set_selected_dataset", "set_labeling_backend", "set_labeling_path"}
                return ALWAYS | {"set_selected_dataset", "export_to_cvat", "set_labeling_backend", "set_labeling_path"}

        if al.labeling_path == LabelingPath.AUTO:
            if not al.auto_labeling_complete:
                # configure_auto_labeling and set_auto_labeling_hyperparams are visible
                # from the start of the auto path; their respective preconditions
                # (can_configure_auto_labeling, can_set_auto_labeling_hyperparams)
                # block premature calls with clear recovery messages.
                base = ALWAYS | {
                    "set_selected_dataset",
                    "configure_auto_labeling",
                    "list_model_sources_and_models",
                    "set_auto_labeling_hyperparams",
                    "set_labeling_backend",
                    "set_labeling_path",
                }
                if al.run_awaiting_confirmation and not al.run_confirmed:
                    return base | {"confirm_run"}
                return base | {"run_auto_labeling"}
            if al.labels_imported:
                return ALWAYS | {"launch_voxel51_session"}
            import_tool = "import_from_label_studio" if ls else "import_from_cvat"
            return ALWAYS | {import_tool}

    @classmethod
    def load(cls) -> "WorkflowState":
        try:
            importlib.reload(_cc)
            raw = dict(_cc.WORKFLOW_STATE)
            raw = cls._migrate(raw)
            state = cls.model_validate(raw)

            # TTL: if config.py has not been written in over an hour, the session
            # that set run_awaiting_confirmation / export_confirmed is stale.
            # Reset those flags so a fresh conversation does not resume a dead gate.
            try:
                import time as _time
                age = _time.time() - CONFIG_PATH.stat().st_mtime
                if age > 3600 and state.auto_labeling:
                    if state.auto_labeling.run_awaiting_confirmation:
                        state.auto_labeling.run_awaiting_confirmation = False
                        logging.warning("[STATE] TTL: cleared stale run_awaiting_confirmation")
                    if state.auto_labeling.export_confirmed:
                        state.auto_labeling.export_confirmed = False
                        logging.warning("[STATE] TTL: cleared stale export_confirmed")
            except Exception:
                pass

            return state
        except Exception as e:
            logging.warning(f"[STATE] Failed to load WorkflowState: {e} — using defaults")
            return cls()

    @classmethod
    def _migrate(cls, raw: dict) -> dict:
        if raw.get("workflow_name") is None:
            raw["workflow_name"] = ""

        old_al_fields = {
            "auto_labeling_complete": "auto_labeling_complete",
            "cvat_task_id": "cvat_task_id",
        }
        wf = raw.get("workflow_name", "")
        if wf == "auto_labeling" and "auto_labeling" not in raw:
            substate = {}
            for old_key, new_key in old_al_fields.items():
                if old_key in raw:
                    val = raw.pop(old_key)
                    if new_key == "cvat_task_id" and val is None:
                        val = 0
                    substate[new_key] = val
            if substate:
                raw["auto_labeling"] = substate
        else:
            for old_key in old_al_fields:
                raw.pop(old_key, None)

        al = raw.get("auto_labeling")
        if isinstance(al, dict):
            al.setdefault("labeling_backend", "")
            al.setdefault("ls_task_ids", [])
            al.setdefault("manual_classes", [])
            al.setdefault("models_listed", False)
            al.setdefault("export_confirmed", False)
            al.setdefault("run_confirmed", False)
            al.setdefault("run_awaiting_confirmation", False)
            al.setdefault("model_source", "")
            al.setdefault("model_name", "")
            al.setdefault("phase", "")
            al.pop("pending_dataset_change", None)

        known = {
            "workflow_name", "dataset_name", "dataset_confirmed",
            "labeled_dataset_name", "auto_labeling",
        }
        for key in list(raw.keys()):
            if key not in known:
                raw.pop(key)

        return raw

    def save(self) -> None:
        src = CONFIG_PATH.read_text()
        tree = ast.parse(src)
        lines = src.splitlines()
        state_dict = self.model_dump()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "WORKFLOW_STATE":
                        start = node.lineno - 1
                        end = node.end_lineno
                        lines[start:end] = [f"WORKFLOW_STATE = {repr(state_dict)}"]
                        CONFIG_PATH.write_text("\n".join(lines).rstrip("\n") + "\n")
                        return
        raise RuntimeError(
            f"WORKFLOW_STATE assignment not found in config.py — save aborted"
        )

    @classmethod
    def reset(cls) -> "WorkflowState":
        fresh = cls()
        fresh.save()
        return fresh

    def reset_for_workflow(self, workflow_name: str) -> "WorkflowState":
        fresh = WorkflowState(workflow_name=workflow_name)
        if workflow_name == "auto_labeling":
            fresh.auto_labeling = AutoLabelingState()
        fresh.workflow_just_reset = True
        fresh.save()
        return fresh

    def current_substate(self):
        return getattr(self, self.workflow_name, None) if self.workflow_name else None