"""MSight-pipeline workflow handlers for ChatPipeline, split out of
chat_pipeline.py to keep that file from growing without bound as more
workflows are added. Mixed into ChatPipeline via multiple inheritance --
methods here run with the same `self` (self.state, self._progress_cb,
self._intro_sent_this_turn, etc.) as methods defined directly on ChatPipeline
or on other handler mixins.
"""
import json
import logging
import time

from host_utils import resolve_host
from mcptools.msight_docker import (
    _calibration_status, _get_msight_path, _reset_calibration_to_default,
    calibration_state_label,
)
from pipeline_common import Sentinels, HardStop, Injection, FallThrough, ToolRouting, unwrap_tool_output
from progress_relay import clear_active_progress_cb, set_active_progress_cb
from validate_workflow_state import MsightPipelineState


class MsightPipelineHandlers:
    """Mixin: all msight_pipeline-workflow tool handlers. See chat_pipeline.ChatPipeline."""

    def _maybe_reset_msight_calibration(self, workflow_name: str) -> None:
        """Per explicit user request: a fresh msight_pipeline entry must not
        inherit a previous session's uploaded calibration -- reset to default."""
        if workflow_name != "msight_pipeline":
            return
        msight_path, err = _get_msight_path()
        if err:
            return
        try:
            _reset_calibration_to_default(msight_path)
            logging.warning("[PIPELINE] msight_pipeline entry: calibration reset to shipped default")
        except Exception as e:
            logging.warning(f"[PIPELINE] Failed to reset calibration to default: {e}")

    async def _handle_confirm_msight_run(self) -> tuple[str, list[ToolRouting]]:
        if self.state.msight_pipeline is None:
            self.state.msight_pipeline = MsightPipelineState()
        mp = self.state.msight_pipeline
        mp.run_confirmed = True
        mp.run_awaiting_confirmation = False
        mp.run_confirmation_requested_at = 0.0
        self.state.save()

        source = f"rtsp_url={mp.rtsp_url!r}" if mp.rtsp_url else f"video_input={mp.video_input!r}"
        logging.warning(f"[PIPELINE] confirm_run (msight_pipeline): run_confirmed=True {source}")

        video_input_arg = repr(mp.video_input) if mp.video_input else "None"
        rtsp_url_arg    = repr(mp.rtsp_url) if mp.rtsp_url else "None"
        sensor_name_arg = repr(mp.sensor_name) if mp.sensor_name else "None"
        result = "Run consent recorded. Call start_msight_pipeline() now with the previously-stated source."
        return result, [Injection(
            f"RUN_CONFIRMED: Consent recorded. Call "
            f"start_msight_pipeline(video_input={video_input_arg}, rtsp_url={rtsp_url_arg}, "
            f"sensor_name={sensor_name_arg}) immediately — do not ask the user anything else first."
        )]

    def _msight_calibration_summary_line(self) -> str:
        """One-line calibration status for the pre-run consent summary --
        same check as chat_server.py's SESSION_STATE hint, so they never disagree."""
        msight_path, err = _get_msight_path()
        if err:
            return "not available (MSIGHT_VISION_PATH not configured on this host)"
        return calibration_state_label(_calibration_status(msight_path)["state"], prefixed=False)

    def _build_msight_run_needs_confirmation_sentinel(self, mp: MsightPipelineState) -> str:
        source = f"rtsp_url={mp.rtsp_url!r}" if mp.rtsp_url else f"video_input={mp.video_input!r}"
        return (
            f"{Sentinels.RUN_NEEDS_CONFIRMATION} "
            f"{source} sensor_name={mp.sensor_name!r} "
            f"calibration={self._msight_calibration_summary_line()!r} "
            f"recording={mp.recording_active} recording_pending={mp.recording_pending} "
            f"archiving={mp.archiving_active} archiving_pending={mp.archiving_pending}"
        )

    def _format_msight_run_confirmation_prompt(
        self, mp: MsightPipelineState, replacing_running: bool = False
    ) -> str:
        """Pre-run consent summary. replacing_running is set when a pipeline
        is already up and this would swap in a different source -- called
        out explicitly so a Demo user isn't confused by an unexpected
        confirmation step, since Demo otherwise never shows one."""
        source_label = f"RTSP stream: {mp.rtsp_url}" if mp.rtsp_url else f"Video file/folder: {mp.video_input}"
        recording = (
            "enabled" if mp.recording_active else
            "will start automatically once the pipeline starts" if mp.recording_pending else
            "not enabled"
        )
        archiving = (
            "enabled" if mp.archiving_active else
            "will start automatically once the pipeline starts" if mp.archiving_pending else
            "not enabled"
        )
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
            f"- **Calibration:** {self._msight_calibration_summary_line()}\n"
            f"- **Recording:** {recording}\n"
            f"- **Archiving:** {archiving}\n\n"
            f"{question}"
        )

    async def _handle_select_msight_mode(self, fn_args: dict) -> tuple[str, list[ToolRouting]]:
        """Records Demo vs 'run your own pipeline' as real state (see MsightPipelineState.mode)."""
        mode = fn_args.get("mode", "")
        if mode not in ("demo", "custom"):
            return (
                f"MSIGHT_MODE_INVALID: '{mode}' — must be 'demo' or 'custom'."
            ), [FallThrough()]
        if self.state.msight_pipeline is None:
            self.state.msight_pipeline = MsightPipelineState()
        self.state.msight_pipeline.mode = mode
        self.state.save()
        logging.warning(f"[PIPELINE] select_msight_mode: mode={mode!r}")
        return f"MSIGHT_MODE_SET: mode={mode!r}", [FallThrough()]

    def _fallback_demo_intro(self, mp: MsightPipelineState) -> str:
        """Static backstop intro used only if the LLM started the Demo pipeline
        without calling send_intro first."""
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
        """Persist the source on success only. Gated on run_confirmed,
        mirroring AutoLabelingState's confirm_run pattern: the first call
        records the request and shows a consent summary; only a second call,
        after confirm_run, actually starts anything. Wraps the call with
        set/clear_active_progress_cb so ctx.log() streams as "log" SSE events during the docker compose call."""
        if not fn_args.get("video_input") and not fn_args.get("rtsp_url"):
            return (
                "MSIGHT_SOURCE_REQUIRED: start_msight_pipeline needs video_input or "
                "rtsp_url — neither was provided. Do not call this tool speculatively "
                "or to prompt the user for a source; ask them for a video file/folder "
                "path or an RTSP stream URL first, then call this again once they've "
                "given one."
            ), [FallThrough()]

        if self.state.msight_pipeline is None:
            self.state.msight_pipeline = MsightPipelineState()
        mp = self.state.msight_pipeline

        # Captured before any mutation below: True here means this call is
        # the second half of an explicit confirm_run() round trip (a prior
        # turn already showed a consent summary), as opposed to a one-shot
        # Demo start with no consent step at all.
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

        # Demo skips the consent round trip -- derived from tracked state, not
        # a per-call LLM flag (testing showed the model wasn't reliably
        # consistent deciding it itself). Exception: replacing an already-
        # running pipeline with a different source still requires
        # confirmation even in Demo -- the model didn't reliably ask about
        # that on its own either (silently replaced, or stopped the old one
        # without asking, in about half of direct-repro runs).
        skip_confirmation = mp.mode == "demo" and not (was_running and source_changed)

        if not mp.run_confirmed and not skip_confirmation:
            mp.run_awaiting_confirmation = True
            mp.run_confirmation_requested_at = time.time()
            self.state.save()
            sentinel = self._build_msight_run_needs_confirmation_sentinel(mp)
            return sentinel, [HardStop(
                self._format_msight_run_confirmation_prompt(mp, replacing_running=was_running and source_changed)
            )]

        # Safety net: the LLM is told to call send_intro before this, but that
        # instruction gets skipped often enough that we guarantee it here too.
        # Skipped for a call arriving already-confirmed (run_confirmed_at_entry)
        # -- that means a consent summary was already shown last turn, so a
        # fresh "what is MSight" explainer here would just be redundant noise
        # right as the pipeline actually starts.
        if (
            skip_confirmation and not run_confirmed_at_entry
            and not self._intro_sent_this_turn and self._progress_cb
        ):
            await self._progress_cb("intro", {"message": self._fallback_demo_intro(mp)})
            self._intro_sent_this_turn = True

        if self._progress_cb:
            set_active_progress_cb(self._progress_cb)
        try:
            result = unwrap_tool_output(await mcp_client.call_tool("start_msight_pipeline", fn_args))
        finally:
            if self._progress_cb:
                clear_active_progress_cb()
        try:
            mp.pipeline_running = json.loads(result).get("status") == "ok"
        except Exception:
            mp.pipeline_running = False
        # Reset regardless of outcome, so the next start asks for consent again.
        mp.run_confirmed = False
        mp.run_awaiting_confirmation = False
        mp.run_confirmation_requested_at = 0.0

        routings: list[ToolRouting] = [FallThrough()]
        if mp.pipeline_running:
            # Fire any recording/archiving request deferred while the pipeline was still starting.
            if mp.recording_pending:
                sensor_arg = repr(mp.sensor_name) if mp.sensor_name else "None"
                routings.append(Injection(
                    "RECORDING_WAS_PENDING: The user asked to start recording before "
                    "the pipeline was running. It's running now — call "
                    f"start_msight_recording(sensor_name={sensor_arg}) immediately, in "
                    "this same reply, before telling the user the pipeline has started."
                ))
            if mp.archiving_pending:
                bucket_arg = repr(mp.archiving_pending_bucket)
                prefix_arg = repr(mp.archiving_pending_prefix) if mp.archiving_pending_prefix else "None"
                routings.append(Injection(
                    "ARCHIVING_WAS_PENDING: The user asked to start archiving before "
                    "the pipeline was running. It's running now — call "
                    f"start_msight_archiving(s3_bucket={bucket_arg}, s3_prefix={prefix_arg}) "
                    "immediately, in this same reply, before telling the user the pipeline has started."
                ))
        self.state.save()
        return result, routings

    async def _handle_stop_msight_pipeline(
        self, fn_args: dict, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        """Only real addition over calling the MCP tool directly: clears
        pipeline_running on success, so SESSION_STATE stops telling the LLM
        (and, via it, the user) that something's still running once it's
        actually been stopped."""
        result = unwrap_tool_output(await mcp_client.call_tool("stop_msight_pipeline", fn_args))
        try:
            ok = json.loads(result).get("status") == "ok"
        except Exception:
            ok = False
        if ok:
            if self.state.msight_pipeline is None:
                self.state.msight_pipeline = MsightPipelineState()
            self.state.msight_pipeline.pipeline_running = False
            self.state.save()
        return result, [FallThrough()]

    async def _handle_msight_record_archive(
        self, fn_name: str, fn_args: dict, mcp_client
    ) -> tuple[str, list[ToolRouting]]:
        """Persist recording_active/archiving_active on success; also handles the deferred-until-pipeline-running path (see MsightPipelineState)."""
        if self.state.msight_pipeline is None:
            self.state.msight_pipeline = MsightPipelineState()
        mp = self.state.msight_pipeline

        if fn_name == "start_msight_recording" and not mp.pipeline_running:
            mp.recording_pending = True
            mp.recording_pending_since = time.time()
            self.state.save()
            logging.warning("[PIPELINE] start_msight_recording deferred -- pipeline not running yet")
            return json.dumps({
                "status": "deferred",
                "message": "Got it — recording will start automatically as soon as the pipeline is running.",
            }), [FallThrough()]

        if fn_name == "start_msight_archiving" and not mp.pipeline_running:
            mp.archiving_pending = True
            mp.archiving_pending_since = time.time()
            mp.archiving_pending_bucket = fn_args.get("s3_bucket", "")
            mp.archiving_pending_prefix = fn_args.get("s3_prefix") or ""
            self.state.save()
            logging.warning("[PIPELINE] start_msight_archiving deferred -- pipeline not running yet")
            return json.dumps({
                "status": "deferred",
                "message": "Got it — archiving will start automatically as soon as the pipeline is running.",
            }), [FallThrough()]

        # Deferred but never launched -- nothing running, so no MCP call needed.
        if fn_name == "stop_msight_recording" and mp.recording_pending and not mp.recording_active:
            mp.recording_pending = False
            mp.recording_pending_since = 0.0
            self.state.save()
            return json.dumps({"status": "ok", "message": "Cancelled — recording had not started yet."}), [FallThrough()]
        if fn_name == "stop_msight_archiving" and mp.archiving_pending and not mp.archiving_active:
            mp.archiving_pending = False
            mp.archiving_pending_since = 0.0
            mp.archiving_pending_bucket = ""
            mp.archiving_pending_prefix = ""
            self.state.save()
            return json.dumps({"status": "ok", "message": "Cancelled — archiving had not started yet."}), [FallThrough()]

        result = unwrap_tool_output(await mcp_client.call_tool(fn_name, fn_args))
        try:
            result_data = json.loads(result)
        except Exception:
            result_data = {}
        ok = result_data.get("status") == "ok"
        if ok:
            if fn_name == "start_msight_recording":
                mp.recording_active = True
                mp.recording_pending = False
                mp.recording_pending_since = 0.0
            elif fn_name == "start_msight_archiving":
                mp.archiving_active = True
                mp.archiving_pending = False
                mp.archiving_pending_since = 0.0
                mp.archiving_pending_bucket = ""
                mp.archiving_pending_prefix = ""
            elif fn_name == "stop_msight_recording":
                mp.recording_active = False
                mp.recording_pending = False
                mp.recording_pending_since = 0.0
                # Resolve into a ready-to-share download URL for the LLM.
                filename = result_data.get("download_filename")
                if filename:
                    result_data["download_url"] = (
                        f"http://{resolve_host()}:8001/msight/download_recording/{filename}"
                    )
                    result = json.dumps(result_data)
            elif fn_name == "stop_msight_archiving":
                mp.archiving_active = False
                mp.archiving_pending = False
                mp.archiving_pending_since = 0.0
                mp.archiving_pending_bucket = ""
                mp.archiving_pending_prefix = ""
            self.state.save()
        return result, [FallThrough()]
