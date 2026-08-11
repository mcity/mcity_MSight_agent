import json
import logging

from fastmcp import Client

from pipeline_common import (
    Sentinels, HardStop, Injection, FallThrough, ToolRouting,
    unwrap_tool_output, TOOL_STATUS_MESSAGES,
)
from pipeline_handlers.auto_labeling import AutoLabelingHandlers
from pipeline_handlers.msight_pipeline import MsightPipelineHandlers
from validate_workflow_state import (
    WorkflowState, AutoLabelingPhase, WORKFLOW_SPECS, validate_tool_input,
)


class ChatPipeline(AutoLabelingHandlers, MsightPipelineHandlers):
    """
    Owns all processing between the HTTP endpoint and the MCP tools.

    Invariant: every tool result is appended to `messages` immediately after
    execution in `_dispatch`, so the OpenAI message history never has an
    assistant tool_call_id without a matching tool result.

    Workflow-specific tool handlers live in the AutoLabelingHandlers and
    MsightPipelineHandlers mixins (pipeline_handlers/) -- this class holds
    the dispatch loop and the handful of handlers that are genuinely
    workflow-agnostic (workflow switching, send_intro, confirm_run's
    routing, dataset-list fetching).
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
        # Set by _handle_send_intro; lets _handle_start_msight_pipeline's Demo path
        # inject a fallback intro if the LLM skipped calling send_intro itself.
        self._intro_sent_this_turn = False

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
        if progress_cb and fn_name not in ("send_reply", "send_intro"):
            status = TOOL_STATUS_MESSAGES.get(fn_name, f"Running {fn_name.replace('_', ' ')}...")
            await progress_cb("status", {"message": status})

        try:
            handlers = self._build_dispatch_table(fn_name, fn_args, mcp_client, messages, progress_cb)
            handler = handlers.get(fn_name)
            if handler is not None:
                result, routings = await handler()
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

    def _build_dispatch_table(self, fn_name, fn_args, mcp_client, messages, progress_cb):
        """Maps tool name -> zero-arg async handler, closing over this call's
        fn_args/mcp_client/messages/progress_cb. Built fresh per call (cheap
        dict of closures) so registering a new tool is one line here instead
        of another branch in a long if/elif chain. Handlers themselves live
        on this class or on the AutoLabelingHandlers/MsightPipelineHandlers
        mixins -- self._handle_x resolves to whichever defines it via MRO."""
        return {
            "send_reply": lambda: self._handle_send_reply(fn_args),
            "send_intro": lambda: self._handle_send_intro(fn_args, progress_cb),
            "confirm_export": lambda: self._handle_confirm_export(),
            "confirm_run": lambda: self._handle_confirm_run(),
            "select_msight_mode": lambda: self._handle_select_msight_mode(fn_args),
            "select_workflow": lambda: self._handle_select_or_switch_workflow(
                fn_name, fn_args, mcp_client, messages
            ),
            "switch_workflow": lambda: self._handle_select_or_switch_workflow(
                fn_name, fn_args, mcp_client, messages
            ),
            "set_selected_dataset": lambda: self._handle_set_selected_dataset(fn_args, mcp_client),
            "list_model_sources_and_models": lambda: self._handle_list_model_sources_and_models(
                fn_args, mcp_client
            ),
            "configure_auto_labeling": lambda: self._handle_configure_auto_labeling_gated(
                fn_args, mcp_client, messages
            ),
            "set_auto_labeling_hyperparams": lambda: self._handle_set_auto_labeling_hyperparams(
                fn_args, mcp_client
            ),
            "set_msight_localization_config": lambda: self._handle_set_msight_localization_config(
                fn_args, mcp_client
            ),
            "export_to_cvat": lambda: self._handle_export_to_cvat(fn_args, mcp_client),
            "run_auto_labeling": lambda: self._dispatch_run_auto_labeling(fn_args, mcp_client, progress_cb),
            "import_from_cvat": lambda: self._handle_import_from_cvat(fn_args, mcp_client),
            "get_labeling_backend": lambda: self._handle_get_labeling_backend(fn_args, mcp_client),
            "set_labeling_path": lambda: self._handle_set_labeling_path(fn_args),
            "set_labeling_backend": lambda: self._handle_set_labeling_backend(fn_args, mcp_client),
            "export_to_label_studio": lambda: self._handle_export_to_label_studio(fn_args, mcp_client),
            "import_from_label_studio": lambda: self._handle_import_from_label_studio(fn_args, mcp_client),
            "launch_voxel51_session": lambda: self._handle_launch_voxel51(fn_args, mcp_client),
            "reset_workflow_state": lambda: self._handle_reset_workflow_state(mcp_client),
            "start_msight_pipeline": lambda: self._handle_start_msight_pipeline(fn_args, mcp_client),
            "stop_msight_pipeline": lambda: self._handle_stop_msight_pipeline(fn_args, mcp_client),
            "start_msight_recording": lambda: self._handle_msight_record_archive(fn_name, fn_args, mcp_client),
            "start_msight_archiving": lambda: self._handle_msight_record_archive(fn_name, fn_args, mcp_client),
            "stop_msight_recording": lambda: self._handle_msight_record_archive(fn_name, fn_args, mcp_client),
            "stop_msight_archiving": lambda: self._handle_msight_record_archive(fn_name, fn_args, mcp_client),
        }

    async def _handle_send_reply(self, fn_args: dict) -> tuple[str, list[ToolRouting]]:
        msg = fn_args.get("message", "")
        src = fn_args.get("source", "")
        content = f"{msg.strip()}\n[source: {src.strip()}]" if src and src.strip() else msg
        return content, [FallThrough()]

    async def _handle_select_or_switch_workflow(
        self, fn_name: str, fn_args: dict, mcp_client, messages: list | None = None
    ) -> tuple[str, list[ToolRouting]]:
        """Reset state for the target workflow, then call the MCP tool."""
        workflow_name = fn_args.get("workflow_name", "")

        if fn_name == "switch_workflow" and workflow_name:
            al = self.state.auto_labeling
            spec = WORKFLOW_SPECS.get(workflow_name)
            same_workflow = workflow_name == self.state.workflow_name

            # Same workflow, dataset step not done yet — guard against a dataset-list
            # loop. Scoped to dataset-requiring workflows so msight_pipeline (which
            # never confirms one) doesn't misfire this when switching away.
            dataset_list_loop = (
                same_workflow and spec is not None and spec.requires_dataset
                and not self.state.dataset_confirmed
            )
            # Non-dataset workflows have no "restart via switch_workflow" pattern
            # (unlike auto_labeling's documented confirm_restart mechanism), so
            # switching to the already-active one is always a misfire — no-op
            # instead of silently wiping substate via reset_for_workflow.
            same_workflow_no_restart_path = (
                same_workflow and spec is not None and not spec.requires_dataset
            )

            if dataset_list_loop:
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

            if same_workflow_no_restart_path:
                logging.warning(
                    f"[PIPELINE] switch_workflow: '{workflow_name}' already active "
                    f"and has no dataset step — no-op instead of resetting"
                )
                result = f"SWITCH_NOOP: Workflow '{workflow_name}' is already active."
                return result, [Injection(
                    f"The workflow '{workflow_name}' is already active. switch_workflow "
                    f"and reset_workflow_state must NOT be called just to continue within "
                    f"it — respond via send_reply using that workflow's own guidance and "
                    f"tools instead."
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
            self._maybe_reset_msight_calibration(workflow_name)
            result = unwrap_tool_output(await mcp_client.call_tool(fn_name, fn_args))
            return result, [await self._post_workflow_select_routing(workflow_name)]

        if fn_name == "switch_workflow":
            self.state = WorkflowState()
            self.state.save()
            result = unwrap_tool_output(await mcp_client.call_tool(fn_name, fn_args))
            return result, [await self._post_workflow_select_routing(workflow_name)]

        self.state = self.state.reset_for_workflow(workflow_name)
        self._maybe_reset_msight_calibration(workflow_name)
        result = unwrap_tool_output(await mcp_client.call_tool(fn_name, fn_args))
        return result, [await self._post_workflow_select_routing(workflow_name)]

    async def _post_workflow_select_routing(self, workflow_name: str) -> "ToolRouting":
        """After select/switch_workflow: dataset-driven workflows get the dataset
        list; others (e.g. msight_pipeline) fall through to their own per-workflow
        prompt. Relies on chat_server.py rebuilding the system prompt every
        iteration so the LLM isn't stuck on the stale pre-selection prompt."""
        spec = WORKFLOW_SPECS.get(workflow_name)
        if spec and not spec.requires_dataset:
            return FallThrough()
        return HardStop(await self._fetch_and_return_dataset_list())

    async def _handle_send_intro(self, fn_args: dict, progress_cb) -> tuple[str, list[ToolRouting]]:
        """Shows an informational message without ending the turn (unlike send_reply),
        so the LLM can explain what it's about to do before the next tool call's own
        progress/log events stream in. Emitted as an "intro" SSE event; the frontend
        (index.html's convertStatusBubbleToIntro) makes the status bubble persistent."""
        message = fn_args.get("message", "")
        if progress_cb:
            await progress_cb("intro", {"message": message})
        self._intro_sent_this_turn = True
        return message, [FallThrough()]

    async def _handle_confirm_run(self) -> tuple[str, list[ToolRouting]]:
        """confirm_run is workflow-agnostic -- branch on the active workflow
        rather than adding a near-duplicate tool per workflow. The two branches
        (auto_labeling vs msight_pipeline) live on their respective handler
        mixins; this is just the routing between them."""
        if self.state.workflow_name == "msight_pipeline":
            return await self._handle_confirm_msight_run()
        return await self._handle_confirm_auto_labeling_run()

    async def _handle_reset_workflow_state(self, mcp_client) -> tuple[str, list[ToolRouting]]:
        result = unwrap_tool_output(await mcp_client.call_tool("reset_workflow_state", {}))
        self.state = WorkflowState()
        self.state.save()
        logging.warning("[PIPELINE] reset_workflow_state: local state cleared and saved")
        return result, [HardStop(result)]

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
