"""MSight-pipeline handlers, mixed into ChatPipeline (shares its `self`)."""
import json
import logging
import time

from msight_agent.shared.host_utils import resolve_host
from msight_agent.msight.runtime.calibration import (
    calibration_label, calibration_mismatch, reset_calibration_to_default,
)
from msight_agent.msight.runtime.paths import get_msight_path
from msight_agent.agent.routing import (
    Sentinels, HardStop, Injection, FallThrough, ToolRouting, tool_status, unwrap_tool_output,
)
from msight_agent.agent.state import MsightPipelineState

_PENDING_LABEL = "will start automatically once the pipeline starts"


def _response(status: str, message: str) -> str:
    return json.dumps({"status": status, "message": message})


def _source_label(mp: MsightPipelineState) -> str:
    return f"rtsp_url={mp.rtsp_url!r}" if mp.rtsp_url else f"video_input={mp.video_input!r}"


def _arg(value) -> str:
    """A Python-call rendering of an optional argument for injected instructions."""
    return repr(value) if value else "None"


def _clear_recording(mp: MsightPipelineState, active: bool) -> None:
    mp.recording_active = active
    mp.recording_pending = False
    mp.recording_pending_since = 0.0


def _clear_archiving(mp: MsightPipelineState, active: bool) -> None:
    mp.archiving_active = active
    mp.archiving_pending = False
    mp.archiving_pending_since = 0.0
    mp.archiving_pending_bucket = ""
    mp.archiving_pending_prefix = ""


class MsightPipelineHandlers:
    """Mixin: all msight_pipeline-workflow tool handlers. See agent/dispatch.py's ChatPipeline."""

    def _msight_state(self) -> MsightPipelineState:
        if self.state.msight_pipeline is None:
            self.state.msight_pipeline = MsightPipelineState()
        return self.state.msight_pipeline

    def _maybe_reset_msight_calibration(self, workflow_name: str) -> None:
        """Reset calibration to the default on fresh entry, so uploads don't carry over."""
        if workflow_name != "msight_pipeline":
            return
        msight_path, err = get_msight_path()
        if err:
            return
        try:
            reset_calibration_to_default(msight_path)
            logging.warning("[PIPELINE] msight_pipeline entry: calibration reset to shipped default")
        except Exception as e:
            logging.warning(f"[PIPELINE] Failed to reset calibration to default: {e}")

    async def _handle_confirm_msight_run(self) -> tuple[str, list[ToolRouting]]:
        mp = self._msight_state()
        # Only record consent for a summary the user actually saw; don't trust the tool filter.
        if not mp.run_awaiting_confirmation:
            logging.warning("[PIPELINE] confirm_run (msight_pipeline) BLOCKED — no summary is pending")
            return Sentinels.CONFIRM_NOT_PENDING, [Injection(
                "confirm_run was blocked: no pre-run summary is pending, so there is nothing for "
                "the user to have approved. Do NOT call confirm_run again. Call "
                "start_msight_pipeline to produce a fresh summary, and wait for the user to approve it."
            )]
        mp.run_confirmed = True
        mp.run_awaiting_confirmation = False
        mp.run_confirmation_requested_at = 0.0
        self.state.save()

        logging.warning(f"[PIPELINE] confirm_run (msight_pipeline): run_confirmed=True {_source_label(mp)}")
        result = "Run consent recorded. Call start_msight_pipeline() now with the previously-stated source."
        return result, [Injection(
            f"RUN_CONFIRMED: Consent recorded. Call "
            f"start_msight_pipeline(video_input={_arg(mp.video_input)}, rtsp_url={_arg(mp.rtsp_url)}, "
            f"sensor_name={_arg(mp.sensor_name)}) immediately — do not ask the user anything else first."
        )]

    def _build_msight_run_needs_confirmation_sentinel(self, mp: MsightPipelineState) -> str:
        return (
            f"{Sentinels.RUN_NEEDS_CONFIRMATION} "
            f"{_source_label(mp)} sensor_name={mp.sensor_name!r} "
            f"calibration={calibration_label(prefixed=False)!r} "
            f"recording={mp.recording_active} recording_pending={mp.recording_pending} "
            f"archiving={mp.archiving_active} archiving_pending={mp.archiving_pending}"
        )

    def _format_msight_run_confirmation_prompt(
        self, mp: MsightPipelineState, replacing_running: bool = False
    ) -> str:
        """Pre-run consent summary; replacing_running notes a source swap on a live pipeline."""
        source_label = f"RTSP stream: {mp.rtsp_url}" if mp.rtsp_url else f"Video file/folder: {mp.video_input}"
        recording = "enabled" if mp.recording_active else _PENDING_LABEL if mp.recording_pending else "not enabled"
        archiving = "enabled" if mp.archiving_active else _PENDING_LABEL if mp.archiving_pending else "not enabled"
        prefix = (
            "A pipeline is already running — starting this will stop and replace it.\n\n"
            if replacing_running else ""
        )
        question = (
            "Shall I stop the current one and start this instead?"
            if replacing_running else
            "Shall I start the pipeline with this configuration?"
        )
        return (
            f"{prefix}Here's a summary of what will be run:\n\n"
            f"- **Source:** {source_label}\n"
            f"- **Sensor name:** {mp.sensor_name or '(default)'}\n"
            f"- **Calibration:** {calibration_label(prefixed=False)}\n"
            f"- **Recording:** {recording}\n"
            f"- **Archiving:** {archiving}\n\n"
            f"{question}"
        )

    def _deferred_record_archive_routings(self, mp: MsightPipelineState) -> list[ToolRouting]:
        """Start recording/archiving that was deferred until the pipeline is running."""
        routings: list[ToolRouting] = [FallThrough()]
        if not mp.pipeline_running:
            return routings
        if mp.recording_pending:
            routings.append(Injection(
                "RECORDING_WAS_PENDING: The user asked to start recording before "
                "the pipeline was running. It's running now — call "
                f"start_msight_recording(sensor_name={_arg(mp.sensor_name)}) immediately, in "
                "this same reply, before telling the user the pipeline has started."
            ))
        if mp.archiving_pending:
            routings.append(Injection(
                "ARCHIVING_WAS_PENDING: The user asked to start archiving before "
                "the pipeline was running. It's running now — call "
                f"start_msight_archiving(s3_bucket={mp.archiving_pending_bucket!r}, "
                f"s3_prefix={_arg(mp.archiving_pending_prefix)}) "
                "immediately, in this same reply, before telling the user the pipeline has started."
            ))
        return routings

    def _require_run_consent(
        self, mp: MsightPipelineState, replacing_running: bool
    ) -> tuple[str, list[ToolRouting]]:
        """End the turn with the confirmation prompt (unconfirmed half of the gate)."""
        sentinel = self._build_msight_run_needs_confirmation_sentinel(mp)
        return sentinel, [HardStop(
            self._format_msight_run_confirmation_prompt(mp, replacing_running=replacing_running)
        )]

    async def _handle_select_msight_mode(self, fn_args: dict) -> tuple[str, list[ToolRouting]]:
        """Records Demo vs 'run your own pipeline' as real state (see MsightPipelineState.mode)."""
        mode = fn_args.get("mode", "")
        if mode not in ("demo", "custom"):
            return (
                f"MSIGHT_MODE_INVALID: '{mode}' — must be 'demo' or 'custom'."
            ), [FallThrough()]
        mp = self._msight_state()
        if mp.mode != mode:
            mp.video_input = ""
            mp.rtsp_url = ""
        mp.mode = mode
        self.state.save()
        logging.warning(f"[PIPELINE] select_msight_mode: mode={mode!r}")
        return f"MSIGHT_MODE_SET: mode={mode!r}", [FallThrough()]

    def _fallback_demo_intro(self, mp: MsightPipelineState) -> str:
        """Fallback intro if the LLM started Demo without send_intro."""
        source = f"RTSP stream {mp.rtsp_url}" if mp.rtsp_url else f"video {mp.video_input}"
        return (
            "MSight Vision is the camera perception module of the MSight roadside "
            "intelligence ecosystem — 2D object detection (RF-DETR here), fisheye "
            "camera localization, multi-camera fusion, and multi-object tracking for "
            "intelligent transportation deployments. This pipeline's flow: a "
            "video/RTSP source publishes frames over Redis pub/sub, the RF-DETR "
            "detector runs live object detection on them, and a web viewer streams "
            "the annotated result (bounding boxes, class labels, confidence scores) "
            f"to a browser. Starting the demo now with {source}."
        )

    async def _handle_start_msight_pipeline(
        self, fn_args: dict, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        """Consent-gated start: first call shows a summary, the call after confirm_run starts it."""
        if not fn_args.get("video_input") and not fn_args.get("rtsp_url"):
            return (
                "MSIGHT_SOURCE_REQUIRED: start_msight_pipeline needs video_input or "
                "rtsp_url — neither was provided. Do not call this tool speculatively "
                "or to prompt the user for a source; ask them for a video file/folder "
                "path or an RTSP stream URL first, then call this again once they've "
                "given one."
            ), [FallThrough()]

        mp = self._msight_state()

        # True if a consent summary was already shown on a previous turn.
        run_confirmed_at_entry = mp.run_confirmed
        was_running = mp.pipeline_running
        old_source = (mp.video_input, mp.rtsp_url)

        if fn_args.get("rtsp_url"):
            mp.rtsp_url = fn_args["rtsp_url"]
            mp.video_input = ""
        elif fn_args.get("video_input"):
            mp.video_input = fn_args["video_input"]
            mp.rtsp_url = ""
        if fn_args.get("sensor_name"):
            mp.sensor_name = fn_args["sensor_name"]

        source_changed = old_source != (mp.video_input, mp.rtsp_url)

        # Demo skips consent (decided from state, not by the LLM), unless it
        # would replace a running pipeline's source.
        skip_confirmation = mp.mode == "demo" and not (was_running and source_changed)

        # Catch a calibration that can't localize this video before asking for consent.
        msight_path, path_err = get_msight_path()
        if not path_err and (mismatch := calibration_mismatch(msight_path, mp.video_input)):
            self.state.save()
            return _response("error", mismatch), [FallThrough()]

        if not mp.run_confirmed and not skip_confirmation:
            mp.run_awaiting_confirmation = True
            mp.run_confirmation_requested_at = time.time()
            self.state.save()
            return self._require_run_consent(mp, replacing_running=was_running and source_changed)

        # Guarantee the intro the LLM often skips (not needed after a consent summary).
        if (
            skip_confirmation and not run_confirmed_at_entry
            and not self._intro_sent_this_turn and self._progress_cb
        ):
            await self._progress_cb("intro", {"message": self._fallback_demo_intro(mp)})
            self._intro_sent_this_turn = True

        result = unwrap_tool_output(await self._call_tool_with_progress(mcp_client, "start_msight_pipeline", fn_args))
        mp.pipeline_running = tool_status(result) == "ok"
        # Reset regardless of outcome, so the next start asks for consent again.
        mp.run_confirmed = False
        mp.run_awaiting_confirmation = False
        mp.run_confirmation_requested_at = 0.0

        routings = self._deferred_record_archive_routings(mp)
        self.state.save()
        return result, routings

    async def _handle_stop_msight_pipeline(
        self, fn_args: dict, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        """Clears pipeline_running on success."""
        result = unwrap_tool_output(await mcp_client.call_tool("stop_msight_pipeline", fn_args))
        if tool_status(result) == "ok":
            mp = self._msight_state()
            mp.pipeline_running = False
            if mp.mode == "demo":
                mp.video_input = ""
                mp.rtsp_url = ""
            self.state.save()
        return result, [FallThrough()]

    async def _handle_msight_record_archive(
        self, fn_name: str, fn_args: dict, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        """Persist recording/archiving flags; defers a start until the pipeline runs."""
        mp = self._msight_state()
        recording = fn_name.endswith("_recording")
        starting = fn_name.startswith("start_")

        # Demo is watch-only; enforced in code, not just the prompt.
        if starting and mp.mode == "demo":
            return _response("error", "Recording/archiving isn't available in Demo mode. Switch to \"run your "
                             "own pipeline\" (select_msight_mode(mode=\"custom\")) to use it."), [FallThrough()]

        what = "recording" if recording else "archiving"
        if starting and not mp.pipeline_running:
            if recording:
                mp.recording_pending, mp.recording_pending_since = True, time.time()
            else:
                mp.archiving_pending, mp.archiving_pending_since = True, time.time()
                mp.archiving_pending_bucket = fn_args.get("s3_bucket", "")
                mp.archiving_pending_prefix = fn_args.get("s3_prefix") or ""
            self.state.save()
            logging.warning(f"[PIPELINE] {fn_name} deferred -- pipeline not running yet")
            return _response("deferred", f"Got it — {what} will start automatically as soon as the "
                             "pipeline is running."), [FallThrough()]

        # Deferred but never launched -- nothing running, so no MCP call needed.
        pending, active = ((mp.recording_pending, mp.recording_active) if recording
                           else (mp.archiving_pending, mp.archiving_active))
        if not starting and pending and not active:
            (_clear_recording if recording else _clear_archiving)(mp, active=False)
            self.state.save()
            return _response("ok", f"Cancelled — {what} had not started yet."), [FallThrough()]

        result = unwrap_tool_output(await mcp_client.call_tool(fn_name, fn_args))
        if tool_status(result) != "ok":
            return result, [FallThrough()]
        (_clear_recording if recording else _clear_archiving)(mp, active=starting)
        if fn_name == "stop_msight_recording" and (filename := json.loads(result).get("download_filename")):
            # Resolve into a ready-to-share download URL for the LLM.
            result = json.dumps({**json.loads(result),
                                 "download_url": f"http://{resolve_host()}:8001/msight/download_recording/{filename}"})
        self.state.save()
        return result, [FallThrough()]
