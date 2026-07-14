import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

from fastmcp import Client

from validate_workflow_state import (
    WorkflowState, AutoLabelingState, validate_tool_input,
    LabelingBackend, AutoLabelingPhase, LabelingPath,
)

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

from dataclasses import dataclass as _dc


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


class Sentinels:
    """Prefix strings returned by MCP tools to signal specific error/state conditions."""
    DATASET_NOT_FOUND         = "DATASET_NOT_FOUND"
    EXPORT_NEEDS_CONFIRMATION = "EXPORT_NEEDS_CONFIRMATION"
    RUN_NEEDS_CONFIRMATION    = "RUN_NEEDS_CONFIRMATION"
    LS_BACKEND_ERROR          = "LS_BACKEND_ERROR"
    LS_AUTH_ERROR             = "LS_AUTH_ERROR"
    LS_CONNECTION_ERROR       = "LS_CONNECTION_ERROR"
    CVAT_TASK_LIMIT_REACHED   = "CVAT_TASK_LIMIT_REACHED"
    CVAT_FORBIDDEN            = "CVAT_FORBIDDEN"
    CVAT_AUTH_ERROR           = "CVAT_AUTH_ERROR"
    CVAT_NOT_FOUND            = "CVAT_NOT_FOUND"
    CVAT_CONNECTION_ERROR     = "CVAT_CONNECTION_ERROR"
    CVAT_TIMEOUT_ERROR        = "CVAT_TIMEOUT_ERROR"
    BACKEND_NOT_SET           = "BACKEND_NOT_SET"
    LS_NO_ANNOTATIONS         = "LS_NO_ANNOTATIONS"


class ChatPipeline:
    """
    Owns all processing between the HTTP endpoint and the MCP tools.

    Invariant: every tool result is appended to `messages` immediately after
    execution in `_dispatch`, so the OpenAI message history never has an
    assistant tool_call_id without a matching tool result.
    """

    def __init__(self, mcp_client: Client, llm):
        self.mcp_client = mcp_client
        self.llm = llm

        self.state: WorkflowState = WorkflowState()

        # MCP tools write these directly to WORKFLOWS in config.py, not to WorkflowState.
        self.auto_labeling_cache = {
            "mode": ["inference"],
            "epochs": 10,
            "early_stop_patience": 5,
            "early_stop_threshold": 0,
            "learning_rate": 5e-5,
            "weight_decay": 0.0001,
            "max_grad_norm": 0.01,
        }
        self._progress_cb = None  # async (event_type: str, data: dict) -> None

    def _set_flag_if_ok(
        self,
        result: str,
        error_sentinels: list[str],
        state_setter,
    ) -> "HardStop | None":
        """Guard a state-flag write against known error sentinels.

        If any sentinel appears in `result` the setter is NOT called and a
        HardStop with the tool's own error text is returned.  When no sentinel
        matches the setter runs and state is persisted.

        Pass an empty list for `error_sentinels` when the underlying MCP tool
        has no documented error sentinel (flag is always set; a sentinel should
        be added on the MCP server side as a follow-up).
        """
        if error_sentinels and any(s in result for s in error_sentinels):
            err = result.split(":", 1)[1].strip() if ":" in result else result
            return HardStop(err)
        state_setter()
        self.state.save()
        return None

    async def run(self, tool_calls: list, messages: list, progress_cb=None) -> tuple[list, str | None]:
        """Process all tool calls for one request. Reloads WorkflowState from config.py each call."""
        self.state = WorkflowState.load()
        self._progress_cb = progress_cb

        # Sync caches from WORKFLOWS so displayed values reflect prior-request changes.
        try:
            import config.config as _cfg
            _wf = _cfg.WORKFLOWS
            al_wf = _wf.get("auto_labeling", {})
            for k in ("mode", "epochs", "early_stop_patience", "early_stop_threshold",
                      "learning_rate", "weight_decay", "max_grad_norm"):
                if k in al_wf:
                    self.auto_labeling_cache[k] = al_wf[k]
        except Exception:
            pass

        tool_results  = []
        all_routings: list[list[ToolRouting]] = []
        logging.warning(
            f"[PIPELINE] Processing {len(tool_calls)} tool call(s): "
            f"{[c.function.name for c in tool_calls]}"
        )

        mcp_client = self.mcp_client
        for call in tool_calls:
            fn_name = call.function.name

            try:
                fn_args = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            ok, err, fn_args = validate_tool_input(fn_name, fn_args)
            if not ok:
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": fn_name,
                    "content": err,
                })
                tool_results.append({
                    "tool_call_id": call.id,
                    "name": fn_name,
                    "fn_args": fn_args,
                    "result": err,
                })
                all_routings.append([FallThrough()])
                logging.warning(f"[PIPELINE] Tool input validation failed for {fn_name}: {err}")
                continue

            result, routings = await self._dispatch(
                fn_name, fn_args, call, mcp_client, messages, progress_cb
            )
            tool_results.append({
                "tool_call_id": call.id,
                "name": fn_name,
                "fn_args": fn_args,
                "result": result,
            })
            all_routings.append(routings)

        # Workflow reset wipes dataset_confirmed; re-apply if both fired in the same batch.
        workflow_reset_this_batch = any(
            r["name"] in ("select_workflow", "switch_workflow")
            for r in tool_results
        )
        dataset_set_this_batch = next(
            (r for r in tool_results if r["name"] == "set_selected_dataset"), None
        )
        if (
            workflow_reset_this_batch
            and dataset_set_this_batch
            and Sentinels.DATASET_NOT_FOUND not in unwrap_tool_output(
                dataset_set_this_batch.get("result", "")
            )
        ):
            dataset_name = dataset_set_this_batch.get("fn_args", {}).get("dataset_name", "")
            if dataset_name and not self.state.dataset_confirmed:
                logging.warning(
                    f"[PIPELINE] Re-applying dataset confirmation for '{dataset_name}' "
                    f"after workflow reset in same batch"
                )
                self.state.dataset_name = dataset_name
                self.state.dataset_confirmed = True
                self.state.save()

        early_reply = self._orchestrate(all_routings, messages)
        return tool_results, early_reply

    async def _dispatch(
        self, fn_name, fn_args, call, mcp_client, messages, progress_cb=None
    ) -> tuple[str, list[ToolRouting]]:
        """Execute one tool call; always appends result to messages before returning."""
        if progress_cb and fn_name != "send_reply":
            status = TOOL_STATUS_MESSAGES.get(fn_name, f"Running {fn_name.replace('_', ' ')}...")
            await progress_cb("status", {"message": status})

        try:
            if fn_name == "send_reply":
                msg = fn_args.get("message", "")
                src = fn_args.get("source", "")
                content = f"{msg.strip()}\n[source: {src.strip()}]" if src and src.strip() else msg
                result, routings = content, [FallThrough()]

            elif fn_name == "confirm_export":
                result, routings = await self._handle_confirm_export()

            elif fn_name == "confirm_run":
                result, routings = await self._handle_confirm_run()

            elif fn_name in ("select_workflow", "switch_workflow"):
                result, routings = await self._handle_select_or_switch_workflow(
                    fn_name, fn_args, mcp_client, messages
                )

            elif fn_name == "set_selected_dataset":
                result, routings = await self._handle_set_selected_dataset(fn_args, mcp_client)

            elif fn_name == "list_model_sources_and_models":
                result, routings = await self._handle_list_model_sources_and_models(
                    fn_args, mcp_client
                )

            elif fn_name == "configure_auto_labeling":
                model_name_raw = fn_args.get("selected_model", "")
                recent_user_text = " ".join(
                    m["content"].lower()
                    for m in messages[-6:]
                    if m.get("role") == "user" and isinstance(m.get("content"), str)
                )
                if model_name_raw and model_name_raw.lower() not in recent_user_text:
                    logging.warning(
                        f"[PIPELINE] configure_auto_labeling blocked: "
                        f"'{model_name_raw}' not found in recent user messages"
                    )
                    result = (
                        f"CONFIGURE_NEEDS_SELECTION: model name '{model_name_raw}' was not "
                        f"found in the user's recent messages — user must explicitly name a model."
                    )
                    routings = [Injection(
                        f"configure_auto_labeling was blocked: '{model_name_raw}' was not typed "
                        f"by the user in their recent messages. The user likely described their use "
                        f"case without naming a specific model. "
                        f"Use send_reply to provide recommendations and ask: "
                        f"'Which model would you like to use?' — wait for the user to name one explicitly."
                    )]
                else:
                    result, routings = await self._handle_configure_auto_labeling(fn_args, mcp_client)

            elif fn_name == "set_auto_labeling_hyperparams":
                result, routings = await self._handle_set_auto_labeling_hyperparams(
                    fn_args, mcp_client
                )

            elif fn_name == "export_to_cvat":
                result, routings = await self._handle_export_to_cvat(fn_args, mcp_client)

            elif fn_name == "run_auto_labeling":
                if progress_cb:
                    result, routings = await self._handle_run_auto_labeling_streaming(
                        fn_args, progress_cb, mcp_client
                    )
                else:
                    result, routings = await self._handle_run_auto_labeling(fn_args, mcp_client)

            elif fn_name == "import_from_cvat":
                result, routings = await self._handle_import_from_cvat(fn_args, mcp_client)

            elif fn_name == "get_labeling_backend":
                result, routings = await self._handle_get_labeling_backend(fn_args, mcp_client)

            elif fn_name == "set_labeling_path":
                result, routings = await self._handle_set_labeling_path(fn_args)

            elif fn_name == "set_labeling_backend":
                result, routings = await self._handle_set_labeling_backend(fn_args, mcp_client)

            elif fn_name == "export_to_label_studio":
                result, routings = await self._handle_export_to_label_studio(fn_args, mcp_client)

            elif fn_name == "import_from_label_studio":
                result, routings = await self._handle_import_from_label_studio(fn_args, mcp_client)

            elif fn_name == "launch_voxel51_session":
                result, routings = await self._handle_launch_voxel51(fn_args, mcp_client)

            elif fn_name == "reset_workflow_state":
                result, routings = await self._handle_reset_workflow_state(mcp_client)

            else:
                logging.warning(f"[PIPELINE] Calling MCP tool: {fn_name} with args: {fn_args}")
                result = unwrap_tool_output(await mcp_client.call_tool(fn_name, fn_args))
                routings = [FallThrough()]

        except Exception as e:
            err_str = str(e)
            if "WORKFLOW_STATE" in err_str or "WorkflowState" in err_str or "config.py" in err_str:
                logging.warning(f"[PIPELINE] State save failed in {fn_name}: {e}")
                result   = f"STATE_SAVE_ERROR: {err_str}"
                routings = [HardStop(
                    "Failed to save session state. Please try your last action again. "
                    "If the problem persists, try restarting the server."
                )]
            else:
                logging.warning(f"[PIPELINE] Exception in {fn_name}: {e}")
                result   = err_str
                routings = [FallThrough()]

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "name": fn_name,
            "content": result,
        })
        return result, routings

    async def _handle_select_or_switch_workflow(
        self, fn_name: str, fn_args: dict, mcp_client, messages: list | None = None
    ) -> tuple[str, list[ToolRouting]]:
        """Reset state for the (only) auto_labeling workflow, then call the MCP tool."""
        workflow_name = fn_args.get("workflow_name", "")

        if fn_name == "switch_workflow" and workflow_name:
            al = self.state.auto_labeling

            # No dataset yet — guard against dataset-list loop.
            if not self.state.dataset_confirmed:
                logging.warning(
                    f"[PIPELINE] switch_workflow: '{workflow_name}' already active "
                    f"with no dataset confirmed — no-op to prevent loop"
                )
                result = (
                    f"SWITCH_NOOP: Workflow '{workflow_name}' is already active "
                    f"and awaiting dataset selection. "
                    f"Call set_selected_dataset with the user's dataset name."
                )
                return result, [Injection(
                    f"The workflow '{workflow_name}' is already active and no dataset "
                    f"has been confirmed yet. If the user provided a dataset name, call "
                    f"set_selected_dataset(dataset_name=<name>) immediately. "
                    f"Do NOT call switch_workflow again."
                )]

            if al and al.phase in (AutoLabelingPhase.ANNOTATING, AutoLabelingPhase.TRAINING) and not fn_args.get("confirm_restart"):
                action = "annotation export" if al.phase == AutoLabelingPhase.ANNOTATING else "auto-labeling run"
                logging.warning(
                    f"[PIPELINE] switch_workflow blocked: phase={al.phase!r}"
                )
                result = (
                    f"SWITCH_LOCKED: Cannot reconfigure mid-{action}. "
                    f"Parameters are locked until the import step completes. "
                    f"If the user explicitly wants to discard all progress and restart "
                    f"from scratch, confirm with them first, then call switch_workflow "
                    f"again with confirm_restart=true."
                )
                msg = result.split("SWITCH_LOCKED: ", 1)[1] if "SWITCH_LOCKED: " in result else result
                return result, [HardStop(msg)]

            logging.warning(
                f"[PIPELINE] switch_workflow full-reset: '{workflow_name}'"
            )
            self.state = self.state.reset_for_workflow(workflow_name)
            result = unwrap_tool_output(await mcp_client.call_tool(fn_name, fn_args))
            return result, [HardStop(await self._fetch_and_return_dataset_list())]

        if fn_name == "switch_workflow":
            self.state = WorkflowState()
            self.state.save()
            result = unwrap_tool_output(await mcp_client.call_tool(fn_name, fn_args))
            return result, [HardStop(await self._fetch_and_return_dataset_list())]

        self.state = self.state.reset_for_workflow(workflow_name)
        result = unwrap_tool_output(await mcp_client.call_tool(fn_name, fn_args))
        return result, [HardStop(await self._fetch_and_return_dataset_list())]

    def _snapshot_initial_backend(self) -> str:
        """Return the current labeling backend before any state modifications."""
        return (self.state.auto_labeling.labeling_backend if self.state.auto_labeling else "") or ""

    def _preserve_zone_a_state_on_switch(self, dataset_name: str) -> tuple[bool, bool]:
        """If this is a mid-session dataset switch, update AutoLabelingState for Zone A.

        Returns (was_dataset_switch, path_was_preserved).
        Zone A (phase == "") preserves all config flags; locked phases preserve only backend.
        """
        if not (self.state.dataset_confirmed and self.state.dataset_name != dataset_name):
            return False, False

        al = self.state.auto_labeling
        if al and not al.phase:
            # Zone A: carry forward config but drop export/run state for the new dataset.
            self.state.auto_labeling = AutoLabelingState(
                labeling_path=al.labeling_path,
                labeling_backend=al.labeling_backend,
                manual_classes=al.manual_classes,
                models_listed=al.models_listed,
                model_configured=al.model_configured,
                model_source=al.model_source,
                model_name=al.model_name,
                hyperparams_confirmed=al.hyperparams_confirmed,
            )
            path_was_preserved = bool(al.labeling_path)
        else:
            # Locked phase or no auto_labeling state: preserve only backend.
            preserved_backend = al.labeling_backend if al else ""
            self.state.auto_labeling = (
                AutoLabelingState(labeling_backend=preserved_backend)
                if self.state.auto_labeling is not None else None
            )
            path_was_preserved = False

        self.state.dataset_name = ""
        self.state.dataset_confirmed = False
        self.state.save()
        return True, path_was_preserved

    async def _retry_dataset_lookup(self, dataset_name: str, mcp_client) -> str:
        """Call set_selected_dataset with exponential backoff on DATASET_NOT_FOUND.

        FiftyOne's registry cache can lag after recent ingestion, so we retry up to
        three times (at 2 s, 4 s, 6 s) before returning the final result.
        """
        raw = await mcp_client.call_tool("set_selected_dataset", {"dataset_name": dataset_name})
        tool_output = unwrap_tool_output(raw)

        if Sentinels.DATASET_NOT_FOUND not in tool_output:
            return tool_output

        logging.warning(f"[PIPELINE] Dataset '{dataset_name}' not found — retrying up to 3 times")
        for attempt, wait in enumerate([2, 4, 6], start=1):
            await asyncio.sleep(wait)
            raw = await mcp_client.call_tool("set_selected_dataset", {"dataset_name": dataset_name})
            tool_output = unwrap_tool_output(raw)
            if Sentinels.DATASET_NOT_FOUND not in tool_output:
                logging.warning(f"[PIPELINE] Dataset '{dataset_name}' found on retry {attempt}")
                break
            logging.warning(
                f"[PIPELINE] Dataset '{dataset_name}' still not found after retry {attempt}"
            )
        return tool_output

    async def _route_after_dataset_confirmed(
        self,
        tool_output: str,
        dataset_name: str,
        initial_backend: str,
        was_dataset_switch: bool,
        path_was_preserved: bool,
        mcp_client,
    ) -> tuple[str, list[ToolRouting]]:
        """Decide routing after set_selected_dataset returns a non-error result."""
        self.state.dataset_name = dataset_name
        self.state.dataset_confirmed = True
        self.state.save()

        al = self.state.auto_labeling
        effective_backend = (al.labeling_backend if al else "") or initial_backend

        if path_was_preserved and al and al.labeling_path:
            backend_label = "Label Studio" if effective_backend == LabelingBackend.LABEL_STUDIO else "CVAT"
            model_info = (
                f" Model '{al.model_name}' ({al.model_source}) is still configured."
                if al.model_configured else ""
            )
            logging.warning(
                f"[PIPELINE] Dataset switched to '{dataset_name}': "
                f"path={al.labeling_path!r} backend={effective_backend!r} preserved"
            )
            return tool_output, [Injection(
                f"DATASET_SWITCHED: dataset is now '{dataset_name}'. "
                f"Labeling path ({al.labeling_path!r}), backend ({backend_label}), "
                f"and all configuration are preserved.{model_info}\n"
                f"Tell the user: 'Switched to **{dataset_name}** — your configuration is preserved.' "
                + ("Then ask if they want to proceed or make changes." if al.model_configured else "")
            )]

        if effective_backend:
            backend_label = "Label Studio" if effective_backend == LabelingBackend.LABEL_STUDIO else "CVAT"
            dataset_note = (
                f"Switched to **{dataset_name}** (backend preserved: {backend_label})."
                if was_dataset_switch
                else f"**{dataset_name}** confirmed."
            )
            backend_info = {
                "cvat_available": effective_backend == LabelingBackend.CVAT,
                "ls_available":   effective_backend == LabelingBackend.LABEL_STUDIO,
                "active_backend": effective_backend,
                "message":        f"Using previously confirmed backend: {effective_backend}.",
                "dataset_note":   dataset_note,
            }
            logging.warning(
                f"[PIPELINE] Backend preserved for '{dataset_name}': {effective_backend!r}"
            )
            return tool_output, self._routing_for_backend(backend_info)

        backend_info = await self._auto_detect_backend(mcp_client)
        if backend_info:
            return tool_output, self._routing_for_backend(backend_info)
        return tool_output, [Injection(f"DATASET_CONFIRMED: {dataset_name}")]

    async def _handle_set_selected_dataset(
        self, fn_args: dict, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        ok, msg = self.state.can_confirm_dataset()
        if not ok:
            return msg, [FallThrough()]

        dataset_name    = fn_args["dataset_name"]
        initial_backend = self._snapshot_initial_backend()

        logging.warning(
            f"[PIPELINE] set_selected_dataset: new={dataset_name!r} "
            f"current={self.state.dataset_name!r} confirmed={self.state.dataset_confirmed} "
            f"backend={initial_backend!r}"
        )

        # No-op: same dataset already confirmed for this session.
        if self.state.dataset_confirmed and self.state.dataset_name == dataset_name:
            logging.warning(
                f"[PIPELINE] set_selected_dataset: '{dataset_name}' already confirmed — no-op"
            )
            routings: list[ToolRouting] = []
            if self.state.workflow_name == "auto_labeling" and initial_backend:
                backend_info = {
                    "cvat_available": initial_backend == LabelingBackend.CVAT,
                    "ls_available":   initial_backend == LabelingBackend.LABEL_STUDIO,
                    "active_backend": initial_backend,
                    "message":        f"Dataset unchanged; backend confirmed: {initial_backend}.",
                }
                routings = self._routing_for_backend(backend_info)
            return f"Dataset '{dataset_name}' is already confirmed. No change needed.", routings

        was_dataset_switch, path_was_preserved = self._preserve_zone_a_state_on_switch(dataset_name)
        tool_output = await self._retry_dataset_lookup(dataset_name, mcp_client)

        if Sentinels.DATASET_NOT_FOUND not in tool_output:
            return await self._route_after_dataset_confirmed(
                tool_output, dataset_name, initial_backend,
                was_dataset_switch, path_was_preserved, mcp_client,
            )
        self.state.dataset_confirmed = False
        return tool_output, [FallThrough()]

    async def _handle_get_labeling_backend(
        self, fn_args: dict, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        """Detect the active backend and return routing for the LABELING_BACKEND gate."""
        result = unwrap_tool_output(await mcp_client.call_tool("get_labeling_backend", fn_args))
        # Read env vars directly — unwrap_tool_output may return a Python repr, not valid JSON.
        try:
            cvat_ok = bool(os.getenv("CVAT_ACCESS_TOKEN", "").strip())
            ls_ok   = bool(os.getenv("LS_TOKEN", "").strip())
            if cvat_ok and ls_ok:
                active = LabelingBackend.BOTH
            elif ls_ok:
                active = LabelingBackend.LABEL_STUDIO
            elif cvat_ok:
                active = LabelingBackend.CVAT
            else:
                active = LabelingBackend.NONE
            if active in (LabelingBackend.CVAT, LabelingBackend.LABEL_STUDIO, LabelingBackend.BOTH):
                if self.state.auto_labeling is None:
                    self.state.auto_labeling = AutoLabelingState()
                self.state.auto_labeling.labeling_backend = active
                self.state.save()
            backend_info = {
                "cvat_available": cvat_ok,
                "ls_available":   ls_ok,
                "active_backend": active,
                "message":        f"Backend detected: {active}.",
            }
            return result, self._routing_for_backend(backend_info)
        except Exception:
            return result, [FallThrough()]

    async def _handle_list_model_sources_and_models(
        self, fn_args: dict, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        """Fetch model list; returns a formatted HardStop display."""
        if not self.state.dataset_confirmed or not self.state.dataset_name:
            msg = (
                "DATASET_NOT_CONFIRMED: A dataset must be confirmed before "
                "selecting a model. Please call set_selected_dataset first."
            )
            return msg, [FallThrough()]

        if self.state.auto_labeling is None:
            self.state.auto_labeling = AutoLabelingState()
        if not self.state.auto_labeling.labeling_path:
            self.state.auto_labeling.labeling_path = LabelingPath.AUTO
            self.state.save()

        result = unwrap_tool_output(await mcp_client.call_tool("list_model_sources_and_models", fn_args))

        if "DATASET_NOT_CONFIRMED" in result:
            return result, [FallThrough()]

        self.state.auto_labeling.models_listed = True
        self.state.save()
        return result, [HardStop(self._format_model_list(result))]

    async def _auto_detect_backend(self, mcp_client) -> dict | None:
        """Detect backend from env vars directly; avoids repr-vs-JSON parse issues in MCP call.

        Note: token presence is treated as token validity here.  An expired or
        scoped-down token will still report the backend as "available" and the
        error will only surface later at export time.  A network round-trip to
        verify validity on every dataset-selection call is not warranted for this
        low-severity case — this is known/accepted behaviour.
        """
        try:
            cvat_ok = bool(os.getenv("CVAT_ACCESS_TOKEN", "").strip())
            ls_ok   = bool(os.getenv("LS_TOKEN", "").strip())

            if cvat_ok and ls_ok:
                # Do not pre-set a backend — the LLM must ask the user to choose.
                active  = LabelingBackend.BOTH
                message = (
                    "Both CVAT and Label Studio credentials are configured. "
                    "Which would you prefer to use?"
                )
            elif ls_ok:
                active  = LabelingBackend.LABEL_STUDIO
                message = "Label Studio credentials found. Using Label Studio for annotation."
            elif cvat_ok:
                active  = LabelingBackend.CVAT
                message = "CVAT credentials found. Using CVAT for annotation."
            else:
                active  = LabelingBackend.NONE
                message = (
                    "No annotation backend credentials found in .env. "
                    "Please add CVAT_ACCESS_TOKEN or LS_TOKEN before continuing."
                )

            backend_info = {
                "cvat_available": cvat_ok,
                "ls_available":   ls_ok,
                "active_backend": active,
                "message":        message,
            }

            # BOTH is a sentinel so the state hint routes to set_labeling_backend next.
            if active in (LabelingBackend.CVAT, LabelingBackend.LABEL_STUDIO, LabelingBackend.BOTH):
                if self.state.auto_labeling is None:
                    self.state.auto_labeling = AutoLabelingState()
                self.state.auto_labeling.labeling_backend = active
                self.state.save()
                logging.warning(f"[PIPELINE] Backend auto-detected and written to state: {active}")
            else:
                logging.warning(f"[PIPELINE] Backend detection: {active} — not writing to state")

            return backend_info
        except Exception as e:
            logging.warning(f"[PIPELINE] _auto_detect_backend failed: {e}")
            return None

    def _routing_for_backend(self, backend_info: dict) -> list[ToolRouting]:
        """Translate backend detection result into routing items."""
        active       = backend_info.get("active_backend", "")
        cvat_ok      = backend_info.get("cvat_available", False)
        ls_ok        = backend_info.get("ls_available", False)
        msg_text     = backend_info.get("message", "")
        dataset_note = backend_info.get("dataset_note", "")
        both         = cvat_ok and ls_ok

        if both:
            instruction = (
                "Backend detection is complete. Both backends are available but NEITHER is confirmed. "
                "You MUST ask the user which backend they prefer (CVAT or Label Studio) "
                "and call set_labeling_backend(backend=<choice>) before proceeding. "
                "Do NOT present labeling paths yet. Do NOT assume CVAT is chosen."
            )
            return [Injection(f"LABELING_BACKEND: {active}. {msg_text} {instruction}")]

        al = self.state.auto_labeling
        backend_label = "Label Studio" if active == LabelingBackend.LABEL_STUDIO else "CVAT"

        if al and al.labeling_path:
            path_label = (
                "Manual Labeling" if al.labeling_path == LabelingPath.MANUAL
                else "Auto Generated Labeling"
            )
            ack = (
                f"The annotation backend has been switched to **{backend_label}**. "
                f"Continuing with **{path_label}**.\n\n"
            )
            if al.labeling_path == LabelingPath.MANUAL:
                if al.manual_classes:
                    classes_s = ", ".join(al.manual_classes)
                    return [HardStop(
                        f"{ack}Your annotation classes are set to: {classes_s}. "
                        f"Let me know when you would like to proceed with "
                        f"exporting to {backend_label}."
                    )]
                fn_export = (
                    "export_to_label_studio"
                    if active == LabelingBackend.LABEL_STUDIO
                    else "export_to_cvat"
                )
                dataset = self.state.dataset_name or "?"
                return [Injection(
                    f"BACKEND_SWITCHED: annotation backend is now {backend_label}. "
                    f"Continuing with Manual Labeling.\n"
                    f"If the user's current message contains annotation class names, "
                    f"extract them and call "
                    f"{fn_export}(dataset_name={dataset!r}, classes=[...]) immediately. "
                    f"Otherwise ask the user for class names."
                )]
            else:  # auto
                if al.model_configured:
                    model_name     = al.model_name or "?"
                    model_source   = al.model_source or "?"
                    hyperparam_block = f"\nCurrent hyperparameters:\n{self._format_hyperparam_block()}"
                    return [Injection(
                        f"BACKEND_SWITCHED: annotation backend is now {backend_label}. "
                        f"Model {model_name!r} ({model_source}) is still configured.{hyperparam_block}\n"
                        f"Tell the user the backend was switched and the model is still configured. "
                        f"If the user's current message also requests a different model, "
                        f"call configure_auto_labeling immediately (it was already batched if so). "
                        f"Otherwise ask if they want to modify hyperparameters or proceed."
                    )]
                elif al.models_listed:
                    step_ctx = (
                        "The model list was already shown. "
                        "Please select a model to continue."
                    )
                else:
                    step_ctx = "I'll fetch the available models for you to choose from."
                return [HardStop(f"{ack}{step_ctx}")]
        else:
            preamble = f"{dataset_note}\n\n" if dataset_note else ""
            return [HardStop(
                f"{preamble}The annotation backend is set to **{backend_label}**.\n\n"
                f"How would you like to annotate your dataset?\n\n"
                f"- **Manual Labeling**: I will export your dataset to {backend_label} "
                f"where you can annotate the images yourself. Once you are done, "
                f"I will import your labels back.\n"
                f"- **Auto Generated Labeling**: I will run a detection model of your "
                f"choice on your dataset to generate predictions automatically, then "
                f"export to {backend_label} so you can review and correct them."
            )]

    async def _handle_configure_auto_labeling(
        self, fn_args: dict, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        if self.state.auto_labeling is None:
            self.state.auto_labeling = AutoLabelingState()

        if self.state.auto_labeling.model_configured:
            self.state.auto_labeling.reset_from_step("model_configured")
            self.state.save()

        ok, msg = self.state.auto_labeling.can_configure_auto_labeling()
        if not ok:
            return msg, [FallThrough()]

        if not self.state.auto_labeling.labeling_path:
            self.state.auto_labeling.labeling_path = LabelingPath.AUTO
            self.state.save()

        result = unwrap_tool_output(await mcp_client.call_tool("configure_auto_labeling", fn_args))
        al = self.state.auto_labeling

        def _set_model_configured():
            al.model_configured = True
            al.hyperparams_confirmed = True
            al.model_source = fn_args.get("selected_source", "")
            al.model_name = fn_args.get("selected_model", "")
            al.run_confirmed = False
            al.run_awaiting_confirmation = False

        stop = self._set_flag_if_ok(result, ["Invalid model"], _set_model_configured)
        if stop:
            return result, [stop]
        model_name   = al.model_name or "?"
        model_source = al.model_source or "?"
        return result, [HardStop(
            f"**{model_name}** ({model_source}) has been configured.\n\n"
            f"Here are the current hyperparameters:\n\n"
            f"{self._format_hyperparam_block()}\n\n"
            f"Would you like to modify any of these hyperparameters, or are you ready to start?"
        )]

    async def _handle_set_auto_labeling_hyperparams(
        self, fn_args: dict, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        if self.state.auto_labeling is None:
            self.state.auto_labeling = AutoLabelingState()

        ok, msg = self.state.auto_labeling.can_set_auto_labeling_hyperparams()
        if not ok:
            return msg, [FallThrough()]

        for k, v in fn_args.items():
            if v is not None:
                self.auto_labeling_cache[k] = v

        result = unwrap_tool_output(
            await mcp_client.call_tool("set_auto_labeling_hyperparams", self.auto_labeling_cache.copy())
        )
        # No error sentinel from this MCP tool -- it always returns a success string.
        self.state.auto_labeling.hyperparams_confirmed = True
        self.state.auto_labeling.run_confirmed = False
        self.state.auto_labeling.run_awaiting_confirmation = False
        self.state.save()
        changed = {k: v for k, v in fn_args.items() if v is not None}
        changed_lines = "\n".join(f"- **{k}**: {v}" for k, v in changed.items())
        return result, [HardStop(
            f"Updated:\n{changed_lines}\n\n"
            f"Current hyperparameters:\n\n"
            f"{self._format_hyperparam_block()}\n\n"
            f"Would you like to change anything else, or are you ready to start?"
        )]

    async def _handle_confirm_export(self) -> tuple[str, list[ToolRouting]]:
        if self.state.auto_labeling is None:
            self.state.auto_labeling = AutoLabelingState()
        self.state.auto_labeling.export_confirmed = True
        self.state.save()
        backend      = self.state.auto_labeling.labeling_backend or LabelingBackend.CVAT
        classes      = self.state.auto_labeling.manual_classes or []
        classes_repr = "[" + ", ".join(repr(c) for c in classes) + "]"
        dataset      = self.state.dataset_name or "?"
        fn           = "export_to_label_studio" if backend == LabelingBackend.LABEL_STUDIO else "export_to_cvat"
        logging.warning(
            f"[PIPELINE] confirm_export: export_confirmed=True "
            f"dataset={dataset!r} backend={backend!r} classes={classes}"
        )
        result = (
            f"Export consent recorded for dataset '{dataset}' → {backend} "
            f"with classes {classes_repr}. "
            f"Call {fn}(dataset_name={dataset!r}, classes={classes_repr}) now."
        )
        return result, [Injection(
            f"EXPORT_CONFIRMED: Consent recorded. "
            f"Call {fn}(dataset_name={dataset!r}, classes={classes_repr}) immediately."
        )]

    def _build_run_needs_confirmation_sentinel(self) -> str:
        al = self.state.auto_labeling
        dataset  = self.state.dataset_name or "?"
        source   = (al.model_source if al else "") or "?"
        model    = (al.model_name   if al else "") or "?"
        backend  = (al.labeling_backend if al else "") or LabelingBackend.CVAT
        backend_label = "Label Studio" if backend == LabelingBackend.LABEL_STUDIO else "CVAT"
        d = self.auto_labeling_cache
        return (
            f"{Sentinels.RUN_NEEDS_CONFIRMATION} "
            f"dataset={dataset!r} source={source!r} model={model!r} "
            f"backend={backend_label!r} "
            f"mode={d.get('mode')} epochs={d.get('epochs')} "
            f"lr={d.get('learning_rate')} wd={d.get('weight_decay')} "
            f"patience={d.get('early_stop_patience')} threshold={d.get('early_stop_threshold')} "
            f"max_grad_norm={d.get('max_grad_norm')}"
        )

    async def _handle_confirm_run(self) -> tuple[str, list[ToolRouting]]:
        if self.state.auto_labeling is None:
            self.state.auto_labeling = AutoLabelingState()
        self.state.auto_labeling.run_confirmed = True
        self.state.auto_labeling.run_awaiting_confirmation = False
        self.state.save()
        model   = self.state.auto_labeling.model_name or "?"
        dataset = self.state.dataset_name or "?"
        logging.warning(
            f"[PIPELINE] confirm_run: run_confirmed=True "
            f"dataset={dataset!r} model={model!r}"
        )
        result = f"Run consent recorded for dataset '{dataset}' with model '{model}'. Call run_auto_labeling() now."
        return result, [Injection("RUN_CONFIRMED: Consent recorded. Call run_auto_labeling() immediately.")]

    def _build_export_confirmation_block(self, fn_args: dict, backend_label: str) -> str:
        """Returns CLASSES_REQUIRED sentinel if no classes, else EXPORT_NEEDS_CONFIRMATION."""
        classes = fn_args.get("classes") or []
        state_classes = (
            self.state.auto_labeling.manual_classes if self.state.auto_labeling else []
        ) or []
        effective_classes = classes or state_classes

        if not effective_classes:
            return "CLASSES_REQUIRED: No annotation classes provided."

        if classes and self.state.auto_labeling:
            self.state.auto_labeling.manual_classes = classes
            self.state.save()
        dataset   = self.state.dataset_name or "?"
        classes_s = ", ".join(effective_classes)
        return (
            f"{Sentinels.EXPORT_NEEDS_CONFIRMATION} "
            f"dataset={dataset!r} backend={backend_label!r} classes=[{classes_s}]"
        )

    async def _handle_export_generic(
        self, backend: str, fn_args: dict, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        """Shared implementation for export_to_cvat and export_to_label_studio.

        Parameterized by backend ("cvat" or "label_studio"); all other per-backend
        differences (env var, tool name, success key, task-state field) are derived
        from that single value so the logic only lives in one place.
        """
        is_ls = backend == LabelingBackend.LABEL_STUDIO
        backend_label = "Label Studio" if is_ls else "CVAT"
        tool_name = "export_to_label_studio" if is_ls else "export_to_cvat"

        if self.state.auto_labeling and not self.state.auto_labeling.labeling_backend:
            msg = (
                f"{Sentinels.BACKEND_NOT_SET}: The annotation backend has not been determined yet. "
                "Please wait — the system will detect your configured backend first."
            )
            return msg, [HardStop(msg.split(":", 1)[1].strip())]

        with_predictions = fn_args.get("with_predictions", False)

        if self.state.workflow_name == "auto_labeling":
            if self.state.auto_labeling is None:
                self.state.auto_labeling = AutoLabelingState()

            if self.state.auto_labeling.labeling_path == LabelingPath.AUTO and not with_predictions:
                msg = (
                    "Export is handled automatically after auto-labeling completes. "
                    "Please run the auto-labeling workflow first."
                )
                return msg, [FallThrough()]

            if not self.state.auto_labeling.labeling_path:
                self.state.auto_labeling.labeling_path = LabelingPath.MANUAL if not with_predictions else LabelingPath.AUTO
                self.state.save()

            ok, msg = (
                self.state.auto_labeling.can_export_to_label_studio(with_predictions)
                if is_ls
                else self.state.auto_labeling.can_export_to_cvat(with_predictions)
            )
            if not ok:
                return msg, [FallThrough()]

        cred_env  = "LS_TOKEN" if is_ls else "CVAT_ACCESS_TOKEN"
        if not os.getenv(cred_env, "").strip():
            msg = (
                f"No {backend_label} credentials found. "
                f"Please add {cred_env} to your .env file and restart the server."
            )
            return msg, [HardStop(msg)]

        al = self.state.auto_labeling
        has_existing_task = bool(al and al.ls_task_ids) if is_ls else bool(al and al.cvat_task_id)
        if not with_predictions and not has_existing_task and not (al and al.export_confirmed):
            sentinel = self._build_export_confirmation_block(fn_args, backend_label)
            if "CLASSES_REQUIRED" in sentinel:
                return sentinel, [HardStop(
                    "What label names would you like to annotate? "
                    "Please provide a list (e.g., car, pedestrian, cyclist)."
                )]
            dataset   = self.state.dataset_name or "?"
            classes_s = ", ".join(al.manual_classes) if (al and al.manual_classes) else "?"
            return sentinel, [HardStop(
                f"Ready to export **{dataset}** to {backend_label} for manual annotation "
                f"with labels: {classes_s}.\n\nShall I proceed?"
            )]

        if not with_predictions and fn_args.get("classes"):
            if self.state.auto_labeling is None:
                self.state.auto_labeling = AutoLabelingState()
            self.state.auto_labeling.manual_classes = fn_args["classes"]
            self.state.save()

        result = unwrap_tool_output(await mcp_client.call_tool(tool_name, fn_args))

        # CVAT-specific error sentinels checked first for CVAT exports.
        if not is_ls and any(s in result for s in [
            Sentinels.CVAT_TASK_LIMIT_REACHED, Sentinels.CVAT_FORBIDDEN, Sentinels.CVAT_AUTH_ERROR,
            Sentinels.CVAT_NOT_FOUND, Sentinels.CVAT_CONNECTION_ERROR, Sentinels.CVAT_TIMEOUT_ERROR,
            "CVAT upload failed:",
        ]):
            err = result.split(":", 1)[1].strip() if ":" in result else result
            return result, [HardStop(err)]

        if any(s in result for s in [
            Sentinels.LS_AUTH_ERROR, Sentinels.LS_CONNECTION_ERROR,
            Sentinels.LS_BACKEND_ERROR, Sentinels.BACKEND_NOT_SET,
        ]):
            err = result.split(":", 1)[1].strip() if ":" in result else result
            return result, [HardStop(err)]

        al = self.state.auto_labeling
        if al:
            if is_ls and "Project ID" in result:
                try:
                    tasks_file = Path(__file__).resolve().parents[1] / "output" / "ls_tasks.json"
                    if tasks_file.exists():
                        registry = json.loads(tasks_file.read_text())
                        dataset_name = fn_args.get("dataset_name", "")
                        if dataset_name in registry:
                            al.ls_task_ids = registry[dataset_name].get("task_ids", [])
                except Exception:
                    pass
                if not al.ls_task_ids:
                    return result, [HardStop(
                        "The export may have succeeded on the Label Studio backend, but the "
                        "task IDs could not be read from the task registry. Please check Label "
                        "Studio directly for your project, or retry the export. If the problem "
                        "persists, contact support."
                    )]
                al.phase = AutoLabelingPhase.ANNOTATING
                self.state.save()
                return result, [HardStop(
                    f"{result.strip()}\n\n"
                    f"Let me know when you have finished annotating and I will import your labels."
                )]
            if not is_ls and "Task ID:" in result:
                try:
                    task_id = int(result.split("Task ID:")[1].split()[0].strip())
                    al.cvat_task_id = task_id
                    al.phase = AutoLabelingPhase.ANNOTATING
                    self.state.save()
                    return result, [HardStop(
                        f"{result.strip()}\n\n"
                        f"Let me know when you have finished annotating and I will import your labels."
                    )]
                except Exception:
                    return result, [HardStop(
                        "The export may have succeeded on the CVAT backend, but the task ID "
                        "could not be read from the response. Please check CVAT directly for "
                        "your task, or retry the export. If the problem persists, contact support."
                    )]

        return result, [FallThrough()]

    async def _handle_export_to_cvat(
        self, fn_args: dict, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        return await self._handle_export_generic(LabelingBackend.CVAT, fn_args, mcp_client)

    async def _handle_run_auto_labeling_streaming(
        self, fn_args: dict, progress_cb, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        """Runs main.py directly (bypassing MCP), streaming stdout/stderr line-by-line."""
        if self.state.auto_labeling is None:
            self.state.auto_labeling = AutoLabelingState()

        ok, msg = self.state.auto_labeling.can_run_auto_labeling(
            self.state.dataset_confirmed, self.state.dataset_name,
        )
        if not ok:
            return msg, [FallThrough()]

        if not self.state.auto_labeling.run_confirmed:
            self.state.auto_labeling.run_awaiting_confirmation = True
            self.state.save()
            sentinel = self._build_run_needs_confirmation_sentinel()
            return sentinel, [HardStop(self._format_run_confirmation_prompt())]

        process = await asyncio.create_subprocess_exec(
            sys.executable, "-u", str(MAIN_PATH),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(MAIN_PATH.parent),
        )

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        async def _read(stream, buf: list[str]) -> None:
            while True:
                raw = await stream.readline()
                if not raw:
                    break
                text = _ANSI_RE.sub("", raw.decode("utf-8", errors="ignore")).rstrip()
                if not text.strip():
                    continue
                buf.append(text)
                await progress_cb("log", {"line": text})


        await asyncio.gather(_read(process.stdout, stdout_lines), _read(process.stderr, stderr_lines))
        await process.wait()

        output       = "\n".join(stdout_lines)
        error_output = "\n".join(stderr_lines)
        combined     = output + "\n" + error_output

        error_lines = [l for l in stderr_lines if " - ERROR - " in l]

        if "Evaluating detections..." in combined:
            res_lines, capture = [], False
            for line in stdout_lines:
                if "              precision    recall  f1-score   support" in line:
                    capture = True
                    res_lines.append(line)
                elif capture and line.startswith("You have launched a remote App on port 5151"):
                    break
                elif capture:
                    res_lines.append(line)
            report = "\n".join(res_lines).strip() or "No inference results found."
        elif error_lines:
            report = "Workflow completed with errors:\n" + "\n".join(error_lines[-10:])
        else:
            report = "Workflow completed successfully.\nThe model is ready to be tested using inference on the validation set."

        log_path = "output/logs/last_auto_labeling_log.txt"
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(
            f"=== STDOUT ===\n{output}\n\n=== STDERR ===\n{error_output}\n\n=== EXIT CODE ===\n{process.returncode}",
            encoding="utf-8",
        )

        if process.returncode == 0:
            result = (
                f"Auto-labeling workflow completed.\n\n"
                f"**Result Summary:**\n```\n{report}\n```\n"
                f"Full logs saved to `{log_path}`"
            )
        else:
            result = (
                f"Auto-labeling failed with exit code {process.returncode}.\n"
                f"Error details:\n```\n{error_output[-3000:]}\n```\n"
                f"Full logs saved to `{log_path}`"
            )
        finalized = await self._finalize_auto_labeling(result, mcp_client)
        return result, [HardStop(finalized)]

    async def _handle_run_auto_labeling(
        self, fn_args: dict, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        if self.state.auto_labeling is None:
            self.state.auto_labeling = AutoLabelingState()

        ok, msg = self.state.auto_labeling.can_run_auto_labeling(
            self.state.dataset_confirmed, self.state.dataset_name,
        )
        if not ok:
            return msg, [FallThrough()]

        if not self.state.auto_labeling.run_confirmed:
            self.state.auto_labeling.run_awaiting_confirmation = True
            self.state.save()
            sentinel = self._build_run_needs_confirmation_sentinel()
            return sentinel, [HardStop(self._format_run_confirmation_prompt())]

        result = unwrap_tool_output(await mcp_client.call_tool("run_auto_labeling", fn_args))
        finalized = await self._finalize_auto_labeling(result, mcp_client)
        return result, [HardStop(finalized)]

    async def _handle_import_from_cvat(
        self, fn_args: dict, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        if self.state.auto_labeling:
            ok, msg = self.state.auto_labeling.can_import_from_cvat()
            if not ok:
                return msg, [FallThrough()]
        result = unwrap_tool_output(await mcp_client.call_tool("import_from_cvat", fn_args))
        return result, [HardStop(self._finalize_import(fn_args, result))]

    async def _handle_set_labeling_backend(
        self, fn_args: dict, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        """Validate credentials for the chosen backend, update state, and persist."""
        requested = fn_args.get("backend", LabelingBackend.CVAT)
        current   = (
            self.state.auto_labeling.labeling_backend if self.state.auto_labeling else ""
        ) or ""

        if current == requested:
            logging.warning(
                f"[PIPELINE] set_labeling_backend: '{requested}' already confirmed — no-op"
            )
            backend_info = {
                "cvat_available": requested == LabelingBackend.CVAT,
                "ls_available":   requested == LabelingBackend.LABEL_STUDIO,
                "active_backend": requested,
                "message":        f"Backend already confirmed: {requested}.",
            }
            return (
                f"Backend is already set to {requested}. No change needed.",
                self._routing_for_backend(backend_info),
            )

        result = unwrap_tool_output(await mcp_client.call_tool("set_labeling_backend", fn_args))
        # LS_BACKEND_ERROR is the sentinel for missing credentials on EITHER backend
        # (the MCP tool uses this prefix regardless of whether CVAT or LS was requested).
        # "Invalid backend" should not appear here due to Pydantic pre-validation but is
        # checked defensively to match the pattern in _handle_export_to_cvat.
        _backend_error_sentinels = [Sentinels.LS_BACKEND_ERROR, "Invalid backend"]
        if any(s in result for s in _backend_error_sentinels):
            err = result.split(":", 1)[1].strip() if ":" in result else result
            return result, [HardStop(err)]
        if self.state.auto_labeling is None:
            self.state.auto_labeling = AutoLabelingState()
        self.state.auto_labeling.labeling_backend = requested
        logging.warning(f"[PIPELINE] set_labeling_backend: {current!r} → {requested!r}")
        self.state.save()
        backend_info = {
            "cvat_available": requested == LabelingBackend.CVAT,
            "ls_available":   requested == LabelingBackend.LABEL_STUDIO,
            "active_backend": requested,
            "message":        f"Backend confirmed: {requested}.",
        }
        return result, self._routing_for_backend(backend_info)

    async def _handle_set_labeling_path(
        self, fn_args: dict
    ) -> tuple[str, list[ToolRouting]]:
        path = fn_args.get("path", "")
        if path not in (LabelingPath.MANUAL, LabelingPath.AUTO):
            msg = f"INVALID_PATH: '{path}' is not a valid labeling path. Must be 'manual' or 'auto'."
            return msg, [FallThrough()]
        if self.state.auto_labeling is None:
            self.state.auto_labeling = AutoLabelingState()
        al = self.state.auto_labeling
        # Mid-Zone-A path switch: reset downstream state but keep dataset and backend.
        if al.labeling_path and al.labeling_path != path:
            preserved_backend = al.labeling_backend
            logging.warning(
                f"[PIPELINE] set_labeling_path: switching {al.labeling_path!r} → {path!r}, "
                f"downstream reset, backend='{preserved_backend}' preserved"
            )
            self.state.auto_labeling = AutoLabelingState()
            self.state.auto_labeling.labeling_backend = preserved_backend
            al = self.state.auto_labeling
        al.labeling_path = path
        self.state.save()
        logging.warning(f"[PIPELINE] set_labeling_path: path={path!r}")
        result = f"Labeling path set to '{path}'."
        al = self.state.auto_labeling
        if path == LabelingPath.MANUAL:
            if al.manual_classes:
                classes_s = ", ".join(al.manual_classes)
                backend_l = "Label Studio" if al.labeling_backend == LabelingBackend.LABEL_STUDIO else "CVAT"
                return result, [HardStop(
                    f"Manual Labeling selected.\n\n"
                    f"Your annotation classes are set to: {classes_s}. "
                    f"Let me know when you would like to proceed with "
                    f"exporting to {backend_l}."
                )]
            fn_export = (
                "export_to_label_studio"
                if al.labeling_backend == LabelingBackend.LABEL_STUDIO
                else "export_to_cvat"
            )
            dataset = self.state.dataset_name or "?"
            return result, [Injection(
                "LABELING_PATH_SET: manual. "
                "Tell the user Manual Labeling has been selected. "
                f"If the user's current message contains annotation class names, "
                f"extract them and call "
                f"{fn_export}(dataset_name={dataset!r}, classes=[...]) immediately. "
                f"Otherwise ask for annotation class names (e.g., car, pedestrian, cyclist)."
            )]
        else:  # auto
            return result, [Injection(
                "LABELING_PATH_SET: auto. "
                "Tell the user Auto Generated Labeling has been selected. "
                "Then proceed to STEP R1: call list_model_sources_and_models() immediately. "
                "Do NOT re-present the labeling path options."
            )]

    async def _handle_export_to_label_studio(
        self, fn_args: dict, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        return await self._handle_export_generic(LabelingBackend.LABEL_STUDIO, fn_args, mcp_client)

    async def _handle_import_from_label_studio(
        self, fn_args: dict, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        if self.state.auto_labeling:
            ok, msg = self.state.auto_labeling.can_import_from_label_studio()
            if not ok:
                return msg, [FallThrough()]
        result = unwrap_tool_output(await mcp_client.call_tool("import_from_label_studio", fn_args))
        if Sentinels.LS_NO_ANNOTATIONS in result:
            err = result.split(":", 1)[1].strip() if ":" in result else result
            return result, [HardStop(err)]
        return result, [HardStop(self._finalize_import(fn_args, result))]

    async def _handle_launch_voxel51(
        self, fn_args: dict, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        dataset_name = fn_args.get("dataset_name", "").strip()
        if not dataset_name:
            dataset_name = self.state.labeled_dataset_name.strip() or self.state.dataset_name.strip()
        if not dataset_name:
            msg = (
                "Cannot launch Voxel51: no dataset name was provided or "
                "found in session state. Please specify the dataset name."
            )
            return msg, [FallThrough()]
        fn_args["dataset_name"] = dataset_name
        logging.warning(f"[PIPELINE] launch_voxel51_session -> dataset='{dataset_name}'")
        result = unwrap_tool_output(await mcp_client.call_tool("launch_voxel51_session", fn_args))
        return result, [FallThrough()]

    async def _handle_reset_workflow_state(self, mcp_client) -> tuple[str, list[ToolRouting]]:
        result = unwrap_tool_output(await mcp_client.call_tool("reset_workflow_state", {}))
        self.state = WorkflowState()
        self.state.save()
        logging.warning("[PIPELINE] reset_workflow_state: local state cleared and saved")
        return result, [HardStop(result)]

    def _format_hyperparam_block(self) -> str:
        """Return the current auto-labeling hyperparameters as a markdown list."""
        d = self.auto_labeling_cache
        return (
            f"- mode: {d.get('mode')}\n"
            f"- epochs: {d.get('epochs')}\n"
            f"- early_stop_patience: {d.get('early_stop_patience')}\n"
            f"- early_stop_threshold: {d.get('early_stop_threshold')}\n"
            f"- learning_rate: {d.get('learning_rate')}\n"
            f"- weight_decay: {d.get('weight_decay')}\n"
            f"- max_grad_norm: {d.get('max_grad_norm')}"
        )

    def _format_run_confirmation_prompt(self) -> str:
        """Format the pre-run confirmation block shown to the user."""
        al           = self.state.auto_labeling
        dataset      = self.state.dataset_name or "?"
        source       = (al.model_source if al else "") or "?"
        model        = (al.model_name   if al else "") or "?"
        backend      = (al.labeling_backend if al else "") or LabelingBackend.CVAT
        backend_label = "Label Studio" if backend == LabelingBackend.LABEL_STUDIO else "CVAT"
        return (
            f"Here's a summary of what will be run:\n\n"
            f"- **Dataset:** {dataset}\n"
            f"- **Model source:** {source}\n"
            f"- **Model:** {model}\n"
            f"- **Annotation backend:** {backend_label}\n"
            f"{self._format_hyperparam_block()}\n\n"
            f"Shall I proceed with auto-labeling?"
        )

    def _orchestrate(self, all_routings: list[list], messages: list) -> str | None:
        """Two-pass: inject all context first, then return last HardStop."""
        if self.state.dataset_confirmed and self.state.dataset_name:
            messages.append({
                "role": "system",
                "content": f"CURRENT_DATASET: {self.state.dataset_name}",
            })

        for routings in all_routings:
            for item in routings:
                if isinstance(item, Injection):
                    messages.append({"role": "system", "content": item.message})

        last_stop: str | None = None
        for routings in all_routings:
            for item in routings:
                if isinstance(item, HardStop):
                    last_stop = item.reply
        return last_stop


    # Reply formatters

    _MODEL_SOURCE_LABELS = {
        "ultralytics":               "Ultralytics",
        "hf_models_objectdetection": "Hugging Face Models for Object Detection",
        "custom_codetr":             "Custom Code Models",
        "roboflow":                  "Roboflow",
    }

    def _format_model_list(self, tool_output: str) -> str:
        try:
            models = json.loads(tool_output)
            lines = []
            counter = 1
            for source_key, model_list in models.items():
                label = self._MODEL_SOURCE_LABELS.get(source_key, source_key)
                lines.append(f"\n**{label}:**")
                for model in model_list:
                    lines.append(f"{counter}. {model}")
                    counter += 1
            return "\n".join(lines) + "\n\nWhich model would you like to use?"
        except Exception:
            return f"{tool_output}\n\nWhich model would you like to use?"

    async def _do_post_run_export(
        self, backend: str, dataset_name: str, mcp_client
    ) -> str:
        """Call the appropriate export tool after auto-labeling and update task-ID state.

        Returns the text to append to the run reply.  Callers catch any exception.
        """
        is_ls = backend == LabelingBackend.LABEL_STUDIO
        tool_name     = "export_to_label_studio" if is_ls else "export_to_cvat"
        backend_label = "Label Studio"            if is_ls else "CVAT"

        export_result = await mcp_client.call_tool(
            tool_name, {"dataset_name": dataset_name, "with_predictions": True}
        )
        export_msg = unwrap_tool_output(export_result)
        success = ("Project ID" in export_msg) if is_ls else ("Task ID:" in export_msg)

        al = self.state.auto_labeling
        if al and success:
            if is_ls:
                try:
                    tasks_file = Path(__file__).resolve().parents[1] / "output" / "ls_tasks.json"
                    if tasks_file.exists():
                        reg = json.loads(tasks_file.read_text())
                        if dataset_name in reg:
                            al.ls_task_ids = reg[dataset_name].get("task_ids", [])
                            self.state.save()
                except Exception:
                    pass
            else:
                try:
                    task_id = int(export_msg.split("Task ID:")[1].split()[0].strip())
                    al.cvat_task_id = task_id
                    self.state.save()
                except Exception:
                    pass

        if not success:
            return (
                f"\n\n{export_msg}"
                f"\n\nExport to {backend_label} did not complete — the predictions were not uploaded. "
                f"Let me know how you'd like to proceed."
            )

        return (
            f"\n\n{export_msg}"
            f"\n\nPlease review and correct the predictions in {backend_label}. "
            f"Let me know when you're done and I'll import the labels back."
        )

    async def _finalize_auto_labeling(self, tool_output: str, mcp_client) -> str:
        al = self.state.auto_labeling
        if not (self.state.dataset_confirmed and al and al.model_configured and al.hyperparams_confirmed):
            return tool_output

        if self.state.auto_labeling:
            self.state.auto_labeling.auto_labeling_complete = True
            self.state.auto_labeling.phase = AutoLabelingPhase.TRAINING
            self.state.save()

        reply = await self._format_auto_labeling_reply(tool_output)
        dataset_name = self.state.dataset_name

        if dataset_name:
            backend = (self.state.auto_labeling.labeling_backend if self.state.auto_labeling else LabelingBackend.CVAT) or LabelingBackend.CVAT

            if self._progress_cb:
                backend_label = "Label Studio" if backend == LabelingBackend.LABEL_STUDIO else "CVAT"
                await self._progress_cb("status", {"message": f"Exporting predictions to {backend_label}..."})

            try:
                reply += await self._do_post_run_export(backend, dataset_name, mcp_client)
            except Exception as e:
                reply += f"\n\nNote: {backend} export failed: {str(e)}"
        else:
            reply += "\n\nNote: Could not determine dataset name for export."

        return reply

    def _finalize_import(self, fn_args: dict, tool_output: str) -> str:
        base_dataset = fn_args.get("dataset_name", "").removesuffix("_labeled")
        labeled_name = f"{base_dataset}_labeled" if base_dataset else ""
        try:
            self.state.labeled_dataset_name = labeled_name
            if self.state.auto_labeling:
                self.state.auto_labeling.labels_imported = True
                self.state.auto_labeling.phase = AutoLabelingPhase.COMPLETE
            self.state.save()
        except Exception:
            pass
        return f"{tool_output.strip()}\n\nWould you like to visualize the labeled dataset in Voxel51?"

    async def _format_auto_labeling_reply(self, tool_output: str) -> str:
        if "precision" in tool_output and "recall" in tool_output and "f1-score" in tool_output:
            summary = await self.llm.summarize_classification_report(tool_output)
            return (
                f"{summary}\n[source: tool result — run_auto_labeling]\n\n"
                f"Full Classification Report:\n"
                f"```\n{tool_output.strip()}\n```"
                f"Would you like to launch Voxel51 to explore the results?"
            )
        return tool_output.strip()

    async def _dataset_not_found_reply(self) -> str:
        try:
            list_result = await self.mcp_client.call_tool("list_datasets", {})
            list_output = unwrap_tool_output(list_result)
            return (
                f"That dataset name wasn't recognized. "
                f"Here are the available datasets:\n\n{list_output}\n\n"
                f"Please select the correct name or re-ingest if needed."
            )
        except Exception as e:
            return f"Dataset not found and couldn't fetch the list: {e}"

    async def _fetch_and_return_dataset_list(self) -> str:
        """Fetch and format the dataset list for display after a workflow switch."""
        try:
            raw = unwrap_tool_output(await self.mcp_client.call_tool("list_datasets", {}))
        except Exception as e:
            return f"Workflow switched. Could not fetch datasets: {e}"
        try:
            datasets = json.loads(raw)
            if isinstance(datasets, list):
                raw = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(datasets))
        except Exception:
            pass
        return (
            f"Here are the available datasets:\n\n{raw}\n\n"
            "Which dataset would you like to use? If you'd like to use your own "
            "dataset, please use the **data ingestion window** on the right to "
            "upload it first (supported formats: raw images, videos, COCO, YOLO, "
            "CVAT-xml)."
        )