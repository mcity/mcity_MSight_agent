"""Auto-labeling handlers, mixed into ChatPipeline (shares its `self`)."""
import asyncio
import json
import logging
import os
import re
import sys
from typing import Optional

from msight_agent.agent.routing import (
    Sentinels, HardStop, Injection, FallThrough, ToolRouting,
    unwrap_tool_output, ANSI_RE,
)
from msight_agent.agent.state import (
    AutoLabelingState, AutoLabelingPhase, LabelingBackend, LabelingPath, config_selected_dataset,
)
from msight_agent.auto_labeling.tools.labeling import summarize_run, write_run_log
from msight_agent.shared.paths import ANNOTATION_TASKS_DIR, MAIN_PATH


# Run tools report failure as "... failed with exit code N", so one without the
# RUN_FAILED prefix is still detected.
_RUN_FAILED_FALLBACK = "failed with exit code"
_LOG_PATH_RE   = re.compile(r"Full logs saved to `([^`]+)`")
_ERROR_LINE_RE = re.compile(r"Last error line:\s*(.+)")

# What the user can change after a failed run. Goes into the reply, so no tool names.
_RETRY_HINT = (
    "the model, or the hyperparameters (mode, epochs, learning rate, "
    "weight decay, early stop patience)"
)


def run_failed(result: str) -> bool:
    """True when a run tool result reports a subprocess that did not succeed."""
    if Sentinels.RUN_FAILED in result:
        return True
    # First line only: a successful run's logs can mention an exit code further down.
    stripped = result.strip()
    first_line = stripped.splitlines()[0] if stripped else ""
    return _RUN_FAILED_FALLBACK in first_line


def export_empty(result: str) -> bool:
    """True when an export uploaded no images -- nothing to annotate, so a failure."""
    return Sentinels.EXPORT_NO_IMAGES in result or "Images: 0\n" in result


def parse_run_failure(result: str) -> tuple[str, str]:
    """(error line, log path) from a failed run tool result."""
    log_match = _LOG_PATH_RE.search(result)
    log_path  = log_match.group(1) if log_match else ""

    err_match = _ERROR_LINE_RE.search(result)
    if err_match:
        return err_match.group(1).strip(), log_path

    # The streaming path returns a fenced stderr block; keep its last real line.
    body  = result.split("Full logs saved to")[0]
    lines = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("```") or line == "Error details:":
            continue
        if _RUN_FAILED_FALLBACK in line:
            continue
        lines.append(line.removeprefix(f"{Sentinels.RUN_FAILED}:").strip())
    return (lines[-1] if lines else "no error output captured"), log_path


# Provenance guards. tool_choice="required" forces a tool call every turn, so an
# ambiguous message ("try again", "ok") used to make the model invent a value and
# configure the session. These refuse a choice the user did not type themselves.
# They fail open: a too-generous token set only lets a real selection through.
# Ordinals count because the labeling paths are shown as a numbered list.
_BACKEND_USER_TOKENS: dict[str, tuple[str, ...]] = {
    LabelingBackend.CVAT:         ("cvat",),
    LabelingBackend.LABEL_STUDIO: ("label studio", "label_studio", "labelstudio", "label-studio"),
}
_PATH_USER_TOKENS: dict[str, tuple[str, ...]] = {
    LabelingPath.MANUAL: ("manual", "manually", "myself", "my own", "by hand", "1", "first"),
    LabelingPath.AUTO:   ("auto", "automatic", "automatically", "auto-generated",
                          "generated", "predictions", "2", "second"),
}


def user_texts(messages: list) -> list[str]:
    """Every user message, lowercased, oldest first (string or text-block content)."""
    texts: list[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            texts.append(content.lower())
        elif isinstance(content, list):
            texts.append(" ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ).lower())
    return texts


def _mentions(text: str, tokens: tuple[str, ...]) -> bool:
    """Whole-word match, so '1' doesn't match inside 'custom_dataset1'."""
    return any(re.search(rf"(?<!\w){re.escape(t)}(?!\w)", text) for t in tokens if t)


def _value_named_by_user(requested: str, choices: dict, texts: list[str]) -> bool:
    """True when the user typed `requested` themselves. The current message counts;
    an earlier one counts only if it names no rival choice (that's a comparison
    question, not a selection)."""
    tokens = choices.get(requested, ())
    if not tokens or not texts:
        return False
    rivals = tuple(t for k, toks in choices.items() if k != requested for t in toks)
    if _mentions(texts[-1], tokens):
        return True
    return any(_mentions(t, tokens) and not _mentions(t, rivals) for t in texts[:-1])


def _free_value_named_by_user(value: str, texts: list[str]) -> bool:
    """Provenance check for free-text values (model names)."""
    return bool(value) and any(_mentions(t, (value.lower(),)) for t in texts)


def _parse_cvat_task_ids(text: str) -> list[int]:
    """Ids from export_to_cvat's "Task IDs: 1, 2, 3" line (large datasets span several tasks)."""
    ids_part = text.split("Task IDs:")[1].split("\n")[0]
    return [int(x.strip()) for x in ids_part.split(",") if x.strip()]


def _backend_label(backend: str) -> str:
    return "Label Studio" if backend == LabelingBackend.LABEL_STUDIO else "CVAT"


def _export_tool(backend: str) -> str:
    return "export_to_label_studio" if backend == LabelingBackend.LABEL_STUDIO else "export_to_cvat"


def _detect_backend_from_env() -> tuple[bool, bool, str]:
    """(cvat token set, LS token set, active backend). Presence only; validity is checked at export."""
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
    return cvat_ok, ls_ok, active


def _ls_task_ids_from_registry(dataset_name: str) -> list[int]:
    """Task ids export_to_label_studio recorded for `dataset_name` ([] if none or unreadable)."""
    tasks_file = ANNOTATION_TASKS_DIR / "ls_tasks.json"
    try:
        registry = json.loads(tasks_file.read_text()) if tasks_file.exists() else {}
    except (OSError, ValueError) as e:
        logging.warning(f"[PIPELINE] Could not read {tasks_file}: {e}")
        return []
    return registry.get(dataset_name, {}).get("task_ids", [])


class AutoLabelingHandlers:
    """Mixin: all auto_labeling-workflow tool handlers. See agent/dispatch.py's ChatPipeline."""

    def _store_backend(self, backend: str) -> None:
        """Persist a detected/chosen backend (BOTH is a sentinel the state hint routes on)."""
        if backend in (LabelingBackend.CVAT, LabelingBackend.LABEL_STUDIO, LabelingBackend.BOTH):
            if self.state.auto_labeling is None:
                self.state.auto_labeling = AutoLabelingState()
            self.state.auto_labeling.labeling_backend = backend
            self.state.save()

    def _handle_run_failure(self, result: str) -> str:
        """Record a failed run and build the reply. No completion flag is set and no
        phase advances, so the workflow stays in Zone A and can be changed and re-run."""
        error, log_path = parse_run_failure(result)
        self.state.record_run("auto_labeling", failed=True, error=error, log_path=log_path)
        attempts = self.state.last_run.attempts if self.state.last_run else 1
        logging.warning(f"[PIPELINE] auto_labeling run failed (attempt {attempts}): {error}")
        log_line = f"\n\nFull logs: `{log_path}`" if log_path else ""
        return (
            f"The auto labeling run did not finish.\n\n"
            f"**Error:** {error}{log_line}\n\n"
            f"Nothing was reset — your settings are still in place and you can "
            f"change {_RETRY_HINT}.\n\n"
            f"Tell me what you would like to change, or say \"run it again\" to "
            f"retry with the same settings."
        )

    def _handle_empty_export(self, backend_label: str, body: str) -> str:
        """Record a run whose export delivered no images (nothing to import) as failed."""
        error = f"the export to {backend_label} contained 0 images"
        self.state.record_run("auto_labeling", failed=True, error=error)
        logging.warning(f"[PIPELINE] auto_labeling: empty export to {backend_label} — run marked failed")
        return (
            f"{body}\n\n"
            f"**The run did not produce anything to annotate — {error}.**\n\n"
            f"Nothing was reset. Please check that the dataset has images, or "
            f"change {_RETRY_HINT} and run it again."
        )

    def _snapshot_initial_backend(self) -> str:
        """Return the current labeling backend before any state modifications."""
        return (self.state.auto_labeling.labeling_backend if self.state.auto_labeling else "") or ""

    def _preserve_zone_a_state_on_switch(self, dataset_name: str) -> tuple[bool, bool]:
        """Handle a mid-session dataset switch. Returns (was_dataset_switch, path_was_preserved)."""
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
        """set_selected_dataset with retries on DATASET_NOT_FOUND (FiftyOne registry lag)."""
        raw = await self._call_tool_with_progress(mcp_client, "set_selected_dataset", {"dataset_name": dataset_name})
        tool_output = unwrap_tool_output(raw)

        if Sentinels.DATASET_NOT_FOUND not in tool_output:
            return tool_output

        logging.warning(f"[PIPELINE] Dataset '{dataset_name}' not found — retrying up to 3 times")
        for attempt, wait in enumerate([2, 4, 6], start=1):
            await asyncio.sleep(wait)
            raw = await self._call_tool_with_progress(mcp_client, "set_selected_dataset", {"dataset_name": dataset_name})
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
            backend_label = _backend_label(effective_backend)
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
            backend_label = _backend_label(effective_backend)
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

        return tool_output, self._routing_for_backend(self._auto_detect_backend())

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

        # No-op: same dataset already confirmed for this session -- and actually
        # selected in config.py (ingest confirms the session state on its own).
        if (self.state.dataset_confirmed and self.state.dataset_name == dataset_name
                and config_selected_dataset() == dataset_name):
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
        cvat_ok, ls_ok, active = _detect_backend_from_env()
        self._store_backend(active)
        backend_info = {
            "cvat_available": cvat_ok,
            "ls_available":   ls_ok,
            "active_backend": active,
            "message":        f"Backend detected: {active}.",
        }
        return result, self._routing_for_backend(backend_info)

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

    _DETECTED_BACKEND_MESSAGES = {
        # BOTH doesn't pre-select a backend — the LLM must ask the user to choose.
        LabelingBackend.BOTH: "Both CVAT and Label Studio credentials are configured. "
                              "Which would you prefer to use?",
        LabelingBackend.LABEL_STUDIO: "Label Studio credentials found. Using Label Studio for annotation.",
        LabelingBackend.CVAT: "CVAT credentials found. Using CVAT for annotation.",
        LabelingBackend.NONE: "No annotation backend credentials found in .env. "
                              "Please add CVAT_ACCESS_TOKEN or LS_TOKEN before continuing.",
    }

    def _auto_detect_backend(self) -> dict:
        """Detect the labeling backend from env vars and store it in state."""
        cvat_ok, ls_ok, active = _detect_backend_from_env()
        self._store_backend(active)
        logging.warning(f"[PIPELINE] Backend auto-detected: {active}")
        return {
            "cvat_available": cvat_ok,
            "ls_available":   ls_ok,
            "active_backend": active,
            "message":        self._DETECTED_BACKEND_MESSAGES[active],
        }

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
        backend_label = _backend_label(active)

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
                fn_export = _export_tool(active)
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

        stop = self._set_flag_if_ok(result, ["Invalid model", "Config not written"], _set_model_configured)
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

    async def _handle_set_msight_localization_config(
        self, fn_args: dict, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        """Set localization_enabled on success (chat tracking only)."""
        if self.state.auto_labeling is None:
            self.state.auto_labeling = AutoLabelingState()

        result = unwrap_tool_output(
            await mcp_client.call_tool("set_msight_localization_config", fn_args)
        )

        if Sentinels.MSIGHT_LOCALIZATION_ERROR in result:
            err = result.split(":", 1)[1].strip() if ":" in result else result
            return result, [HardStop(err)]

        self.state.auto_labeling.localization_enabled = fn_args.get("enabled", True)
        self.state.save()
        return result, [HardStop(
            f"{result}\n\nReady to start, or would you like to change anything else first?"
        )]

    async def _handle_confirm_export(self) -> tuple[str, list[ToolRouting]]:
        if self.state.auto_labeling is None:
            self.state.auto_labeling = AutoLabelingState()

        # Only record consent for a summary the user actually saw; don't trust the tool filter.
        if not self.state.auto_labeling.export_awaiting_confirmation:
            logging.warning("[PIPELINE] confirm_export BLOCKED — no export summary is pending")
            return Sentinels.CONFIRM_NOT_PENDING, [Injection(
                "confirm_export was blocked: no export confirmation summary is pending, "
                "so there is nothing for the user to have approved. Do NOT call "
                "confirm_export again. Call the export tool to produce a fresh summary, "
                "and wait for the user to approve it."
            )]

        self.state.auto_labeling.export_confirmed = True
        self.state.auto_labeling.export_awaiting_confirmation = False
        self.state.save()
        backend      = self.state.auto_labeling.labeling_backend or LabelingBackend.CVAT
        classes      = self.state.auto_labeling.manual_classes or []
        classes_repr = "[" + ", ".join(repr(c) for c in classes) + "]"
        dataset      = self.state.dataset_name or "?"
        fn           = _export_tool(backend)
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
        backend_label = _backend_label(al.labeling_backend if al else "")
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

    def _build_export_confirmation_block(self, fn_args: dict, backend_label: str) -> str:
        """Returns CLASSES_REQUIRED sentinel if no classes, else EXPORT_NEEDS_CONFIRMATION."""
        classes = fn_args.get("classes") or []
        state_classes = (
            self.state.auto_labeling.manual_classes if self.state.auto_labeling else []
        ) or []
        effective_classes = classes or state_classes

        if not effective_classes:
            return "CLASSES_REQUIRED: No annotation classes provided."

        if self.state.auto_labeling:
            if classes:
                self.state.auto_labeling.manual_classes = classes
            # The summary below is being shown, so confirm_export may now record consent.
            self.state.auto_labeling.export_awaiting_confirmation = True
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
        """Shared export_to_cvat / export_to_label_studio implementation."""
        is_ls = backend == LabelingBackend.LABEL_STUDIO
        backend_label = _backend_label(backend)
        tool_name = _export_tool(backend)

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
        has_existing_task = bool(al and al.ls_task_ids) if is_ls else bool(al and al.cvat_task_ids)
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

        result = unwrap_tool_output(await self._call_tool_with_progress(mcp_client, tool_name, fn_args))

        # 0 images exported: no task ids recorded and the phase doesn't advance (stays Zone A).
        if export_empty(result):
            detail = result.split(":", 1)[1].strip() if ":" in result else result
            logging.warning(f"[PIPELINE] {tool_name}: empty export — {detail}")
            return result, [HardStop(
                f"{detail}\n\n"
                f"Nothing was exported to {backend_label}, so there is nothing to annotate. "
                f"Please check that the dataset has images, or select a different dataset."
            )]

        error_routing = self._export_error_routing(result, is_ls)
        if error_routing is not None:
            return result, error_routing

        return self._handle_export_success(result, fn_args, is_ls)

    def _export_error_routing(self, result: str, is_ls: bool) -> Optional[list[ToolRouting]]:
        """None if `result` isn't an error, else routing that ends the turn with it."""
        if not is_ls and any(s in result for s in [
            Sentinels.CVAT_TASK_LIMIT_REACHED, Sentinels.CVAT_STORAGE_LIMIT_REACHED,
            Sentinels.CVAT_FORBIDDEN, Sentinels.CVAT_AUTH_ERROR,
            Sentinels.CVAT_NOT_FOUND, Sentinels.CVAT_CONNECTION_ERROR, Sentinels.CVAT_TIMEOUT_ERROR,
            "CVAT upload failed:",
        ]):
            err = result.split(":", 1)[1].strip() if ":" in result else result
            return [HardStop(err)]

        if any(s in result for s in [
            Sentinels.LS_AUTH_ERROR, Sentinels.LS_CONNECTION_ERROR,
            Sentinels.LS_BACKEND_ERROR, Sentinels.BACKEND_NOT_SET,
        ]):
            err = result.split(":", 1)[1].strip() if ":" in result else result
            return [HardStop(err)]

        return None

    def _handle_export_success(
        self, result: str, fn_args: dict, is_ls: bool
    ) -> tuple[str, list[ToolRouting]]:
        """Record new task id(s) and move to the ANNOTATING phase."""
        al = self.state.auto_labeling
        if not al:
            return result, [FallThrough()]

        if is_ls and "Project ID" in result:
            al.ls_task_ids = _ls_task_ids_from_registry(fn_args.get("dataset_name", "")) or al.ls_task_ids
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

        if not is_ls and "Task IDs:" in result:
            try:
                al.cvat_task_ids = _parse_cvat_task_ids(result)
                al.phase = AutoLabelingPhase.ANNOTATING
                self.state.save()
                return result, [HardStop(
                    f"{result.strip()}\n\n"
                    f"Let me know when you have finished annotating and I will import your labels."
                )]
            except Exception:
                return result, [HardStop(
                    "The export may have succeeded on the CVAT backend, but the task ID(s) "
                    "could not be read from the response. Please check CVAT directly for "
                    "your task(s), or retry the export. If the problem persists, contact support."
                )]

        return result, [FallThrough()]

    async def _handle_export_to_cvat(
        self, fn_args: dict, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        return await self._handle_export_generic(LabelingBackend.CVAT, fn_args, mcp_client)

    def _run_gate(self) -> Optional[tuple[str, list[ToolRouting]]]:
        """Preconditions + consent for a run: None when it may start, else the response."""
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
        return None

    async def _handle_run_auto_labeling_streaming(
        self, progress_cb, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        """Runs main.py directly (bypassing MCP), streaming stdout/stderr line-by-line."""
        if blocked := self._run_gate():
            return blocked

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
                text = ANSI_RE.sub("", raw.decode("utf-8", errors="ignore")).rstrip()
                if not text.strip():
                    continue
                buf.append(text)
                await progress_cb("log", {"line": text})

        await asyncio.gather(_read(process.stdout, stdout_lines), _read(process.stderr, stderr_lines))
        await process.wait()

        output, error_output = "\n".join(stdout_lines), "\n".join(stderr_lines)
        log_path = write_run_log(output, error_output, process.returncode)
        result = summarize_run(output, error_output, process.returncode, log_path)
        finalized = await self._finalize_auto_labeling(result, mcp_client)
        return result, [HardStop(finalized)]

    async def _handle_run_auto_labeling(
        self, fn_args: dict, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        if blocked := self._run_gate():
            return blocked

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
        result = unwrap_tool_output(await self._call_tool_with_progress(mcp_client, "import_from_cvat", fn_args))
        return result, [HardStop(self._finalize_import(fn_args, result))]

    async def _handle_set_labeling_backend(
        self, fn_args: dict, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        """Validate credentials for the chosen backend, update state, and persist."""
        requested = fn_args.get("backend", "")
        current   = (
            self.state.auto_labeling.labeling_backend if self.state.auto_labeling else ""
        ) or ""

        # First selection only accepts a backend the user named; later Zone A changes are exempt.
        if requested not in _BACKEND_USER_TOKENS or current in ("", LabelingBackend.BOTH):
            if not _value_named_by_user(requested, _BACKEND_USER_TOKENS, self._user_texts):
                last = self._user_texts[-1] if self._user_texts else ""
                logging.warning(
                    f"[PIPELINE] set_labeling_backend({requested!r}) BLOCKED — "
                    f"never named by the user; last message {last[:80]!r}"
                )
                return Sentinels.BACKEND_NOT_NAMED, [HardStop(
                    "Which annotation tool would you like to use?\n\n"
                    "1. CVAT\n"
                    "2. Label Studio\n\n"
                    "Please name one and I will set it up."
                )]

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
        # LS_BACKEND_ERROR covers missing credentials for either backend.
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

        if not _value_named_by_user(path, _PATH_USER_TOKENS, self._user_texts):
            last = self._user_texts[-1] if self._user_texts else ""
            logging.warning(
                f"[PIPELINE] set_labeling_path({path!r}) BLOCKED — "
                f"never named by the user; last message {last[:80]!r}"
            )
            return Sentinels.PATH_NOT_NAMED, [HardStop(
                "How would you like to label this dataset?\n\n"
                "1. **Manual Labeling** — I export your dataset to the annotation "
                "tool, you annotate the images, then I import your labels back.\n"
                "2. **Auto Generated Labeling** — I run a detection model of your "
                "choice to generate predictions, then export them so you can review "
                "and correct them.\n\n"
                "Please name one and I will set it up."
            )]

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
                backend_l = _backend_label(al.labeling_backend)
                return result, [HardStop(
                    f"Manual Labeling selected.\n\n"
                    f"Your annotation classes are set to: {classes_s}. "
                    f"Let me know when you would like to proceed with "
                    f"exporting to {backend_l}."
                )]
            fn_export = _export_tool(al.labeling_backend)
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
        result = unwrap_tool_output(await self._call_tool_with_progress(mcp_client, "import_from_label_studio", fn_args))
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
        backend_label = _backend_label(al.labeling_backend if al else "")
        return (
            f"Here's a summary of what will be run:\n\n"
            f"- **Dataset:** {dataset}\n"
            f"- **Model source:** {source}\n"
            f"- **Model:** {model}\n"
            f"- **Annotation backend:** {backend_label}\n"
            f"{self._format_hyperparam_block()}\n\n"
            f"Shall I proceed with auto-labeling?"
        )

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
    ) -> tuple[str, bool]:
        """Export after auto-labeling. Returns (text to append to the run reply, images
        reached the annotation tool); False makes the caller treat the run as failed."""
        is_ls = backend == LabelingBackend.LABEL_STUDIO
        tool_name, backend_label = _export_tool(backend), _backend_label(backend)

        export_result = await self._call_tool_with_progress(
            mcp_client, tool_name, {"dataset_name": dataset_name, "with_predictions": True}
        )
        export_msg = unwrap_tool_output(export_result)
        success = ("Project ID" in export_msg) if is_ls else ("Task IDs:" in export_msg)
        if export_empty(export_msg):
            success = False

        al = self.state.auto_labeling
        if al and success:
            if is_ls:
                al.ls_task_ids = _ls_task_ids_from_registry(dataset_name) or al.ls_task_ids
            else:
                try:
                    al.cvat_task_ids = _parse_cvat_task_ids(export_msg)
                except ValueError as e:
                    logging.warning(f"[PIPELINE] Could not parse CVAT task ids: {e}")
            self.state.save()

        if not success:
            return (
                f"\n\n{export_msg}"
                f"\n\nExport to {backend_label} did not complete — the predictions were not uploaded.",
                False,
            )

        return (
            f"\n\n{export_msg}"
            f"\n\nPlease review and correct the predictions in {backend_label}. "
            f"Let me know when you're done and I'll import the labels back.",
            True,
        )

    async def _finalize_auto_labeling(self, tool_output: str, mcp_client) -> str:
        al = self.state.auto_labeling

        if run_failed(tool_output):
            # Completion flag and phase stay untouched (Zone A) and the post-run export is
            # skipped. Clearing the run confirmation shows a fresh pre-run summary on retry.
            if al:
                al.run_confirmed = False
                al.run_awaiting_confirmation = False
            return self._handle_run_failure(tool_output)

        if not (self.state.dataset_confirmed and al and al.model_configured and al.hyperparams_confirmed):
            self.state.record_run("auto_labeling", failed=False)
            return tool_output

        # auto_labeling_complete gates the with_predictions export below.
        al.auto_labeling_complete = True
        reply = await self._format_auto_labeling_reply(tool_output)
        dataset_name = self.state.dataset_name

        backend = al.labeling_backend or LabelingBackend.CVAT
        backend_label = _backend_label(backend)

        if dataset_name:
            if self._progress_cb:
                await self._progress_cb("status", {"message": f"Exporting predictions to {backend_label}..."})
            try:
                export_note, exported = await self._do_post_run_export(backend, dataset_name, mcp_client)
            except Exception as e:
                logging.warning(f"[PIPELINE] Post-run export to {backend_label} raised: {e}")
                export_note, exported = f"\n\nNote: {backend_label} export failed: {e}", False
        else:
            export_note, exported = "\n\nNote: Could not determine dataset name for export.", False

        # Lock only once predictions reached the annotation tool; otherwise the run
        # counts as failed and the workflow stays reconfigurable.
        if not exported:
            al.auto_labeling_complete = False
            al.phase = AutoLabelingPhase.PENDING
            al.run_confirmed = False
            al.run_awaiting_confirmation = False
            return self._handle_empty_export(backend_label, reply + export_note)

        al.phase = AutoLabelingPhase.TRAINING
        self.state.record_run("auto_labeling", failed=False)
        return reply + export_note

    def _finalize_import(self, fn_args: dict, tool_output: str) -> str:
        base_dataset = fn_args.get("dataset_name", "").removesuffix("_labeled")
        labeled_name = f"{base_dataset}_labeled" if base_dataset else ""
        self.state.labeled_dataset_name = labeled_name
        if self.state.auto_labeling:
            self.state.auto_labeling.labels_imported = True
            self.state.auto_labeling.phase = AutoLabelingPhase.COMPLETE
        self.state.save()
        return f"{tool_output.strip()}\n\nWould you like to visualize the labeled dataset in Voxel51?"

    async def _format_auto_labeling_reply(self, tool_output: str) -> str:
        if "precision" in tool_output and "recall" in tool_output and "f1-score" in tool_output:
            summary = await self.llm.summarize_classification_report(tool_output)
            return (
                f"{summary}\n[source: tool result — run_auto_labeling]\n\n"
                f"Full Classification Report:\n"
                f"```\n{tool_output.strip()}\n```\n\n"
                f"Would you like to launch Voxel51 to explore the results?"
            )
        return tool_output.strip()

    async def _handle_confirm_auto_labeling_run(self) -> tuple[str, list[ToolRouting]]:
        """auto_labeling branch of confirm_run."""
        if self.state.auto_labeling is None:
            self.state.auto_labeling = AutoLabelingState()

        # Last checkpoint before a run that locks the workflow: only record consent for
        # a summary the user actually saw; don't trust the tool filter.
        if not self.state.auto_labeling.run_awaiting_confirmation:
            logging.warning("[PIPELINE] confirm_run BLOCKED — no pre-run summary is pending")
            return Sentinels.CONFIRM_NOT_PENDING, [Injection(
                "confirm_run was blocked: no pre-run confirmation summary is pending, "
                "so there is nothing for the user to have approved. Do NOT call "
                "confirm_run again. Call run_auto_labeling() to produce a fresh "
                "summary, and wait for the user to approve it."
            )]
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

    async def _handle_configure_auto_labeling_gated(
        self, fn_args: dict, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        """Block configure_auto_labeling unless the user actually named the model."""
        model_name_raw = fn_args.get("selected_model", "")
        if model_name_raw and not _free_value_named_by_user(model_name_raw, self._user_texts):
            logging.warning(
                f"[PIPELINE] configure_auto_labeling blocked: "
                f"'{model_name_raw}' not found in recent user messages"
            )
            result = (
                f"CONFIGURE_NEEDS_SELECTION: model name '{model_name_raw}' was not "
                f"found in the user's recent messages — user must explicitly name a model."
            )
            return result, [Injection(
                f"configure_auto_labeling was blocked: '{model_name_raw}' was not typed "
                f"by the user in their recent messages. The user likely described their use "
                f"case without naming a specific model. "
                f"Use send_reply to provide recommendations and ask: "
                f"'Which model would you like to use?' — wait for the user to name one explicitly."
            )]
        return await self._handle_configure_auto_labeling(fn_args, mcp_client)

    async def _dispatch_run_auto_labeling(
        self, fn_args: dict, mcp_client, progress_cb
    ) -> tuple[str, list[ToolRouting]]:
        """Run auto-labeling, streaming progress when a progress_cb is available."""
        if progress_cb:
            return await self._handle_run_auto_labeling_streaming(progress_cb, mcp_client)
        return await self._handle_run_auto_labeling(fn_args, mcp_client)
