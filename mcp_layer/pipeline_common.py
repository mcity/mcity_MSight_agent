"""Shared routing/sentinel primitives used by ChatPipeline and its per-workflow
handler mixins (pipeline_handlers/). Kept as a separate leaf module so those
mixins can import Sentinels/HardStop/etc. without a circular import back to
chat_pipeline.py, which imports the mixins.
"""
import re
from dataclasses import dataclass as _dc
from pathlib import Path

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


class Sentinels:
    """Prefix strings returned by MCP tools to signal specific error/state conditions."""
    DATASET_NOT_FOUND         = "DATASET_NOT_FOUND"
    EXPORT_NEEDS_CONFIRMATION = "EXPORT_NEEDS_CONFIRMATION"
    RUN_NEEDS_CONFIRMATION    = "RUN_NEEDS_CONFIRMATION"
    RESET_NEEDS_CONFIRMATION  = "RESET_NEEDS_CONFIRMATION"
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
    RUN_FAILED                = "RUN_FAILED"
    EXPORT_NO_IMAGES          = "EXPORT_NO_IMAGES"
    BACKEND_NOT_NAMED         = "BACKEND_NOT_NAMED"
    PATH_NOT_NAMED            = "PATH_NOT_NAMED"
    CONFIRM_NOT_PENDING       = "CONFIRM_NOT_PENDING"


# ---------------------------------------------------------------------------
# Provenance guards
#
# chat_server sends tool_choice="required" on the first agentic iteration, so
# the model MUST emit a tool call on every turn. When the user's message is
# ambiguous ("try again", "ok"), that pressure used to make the model invent an
# argument value and silently configure the session. These guards refuse any
# configuration value the user did not type themselves.
#
# They fail OPEN: a token set that is too generous only lets a real selection
# through, while the ambiguous-retry case stays blocked.
# ---------------------------------------------------------------------------

# The per-choice token maps live in pipeline_handlers/auto_labeling.py, keyed by
# the LabelingBackend/LabelingPath constants themselves. Keeping them there
# avoids duplicating those string values in this module, which is deliberately a
# leaf and does not import validate_workflow_state.


def user_texts(messages: list) -> list[str]:
    """Return every user message as lowercased text, oldest first.

    Content is normally a plain string, but list-of-blocks content is handled
    too so the guards still work if the message format changes.
    """
    texts: list[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            texts.append(content.lower())
        elif isinstance(content, list):
            texts.append(" ".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ).lower())
    return texts


def mentions(text: str, tokens: tuple[str, ...]) -> bool:
    """True when `text` contains any token as a whole word.

    Whole-word matching keeps short tokens safe: the ordinal '1' must not match
    inside 'custom_dataset1', and a model name must not match a longer name.
    """
    return any(
        re.search(rf"(?<!\w){re.escape(t)}(?!\w)", text)
        for t in tokens if t
    )


def value_named_by_user(requested: str, choices: dict, texts: list[str]) -> bool:
    """True when the user typed `requested` themselves — never inferred.

    The current message is the primary source. Earlier messages are accepted as
    a fallback, because base_prompt tells the agent not to re-ask for something
    the user already said ("use custom_dataset8 with CVAT"). An earlier message
    that names a RIVAL choice too is not a selection — it is a comparison
    question ("what is the difference between CVAT and Label Studio?") — so it
    does not count.
    """
    tokens = choices.get(requested, ())
    if not tokens or not texts:
        return False
    rivals = tuple(t for k, toks in choices.items() if k != requested for t in toks)
    if mentions(texts[-1], tokens):
        return True
    return any(
        mentions(text, tokens) and not mentions(text, rivals)
        for text in texts[:-1]
    )


def free_value_named_by_user(value: str, texts: list[str]) -> bool:
    """Provenance check for free-text values (model names) with no fixed choice set."""
    if not value:
        return False
    return any(mentions(text, (value.lower(),)) for text in texts)


# Every run_* tool formats its failure return as "... failed with exit code N",
# so a tool that has not been given the RUN_FAILED prefix is still detected.
_RUN_FAILED_FALLBACK = "failed with exit code"

_LOG_PATH_RE   = re.compile(r"Full logs saved to `([^`]+)`")
_ERROR_LINE_RE = re.compile(r"Last error line:\s*(.+)")

# Matches the image-count line of both export summaries: CVAT writes
# "Images: 0", Label Studio pads to "Images     : 0".
_ZERO_IMAGES_RE = re.compile(r"^Images\s*:\s*0\s*$", re.MULTILINE)


def run_failed(result: str) -> bool:
    """True when a run_* tool result reports a subprocess that did not succeed."""
    if Sentinels.RUN_FAILED in result:
        return True
    # Fallback for a tool that has no sentinel yet. Only the first line is
    # tested: a successful run whose captured logs mention an exit code deeper
    # in the report must not be read as a failure.
    stripped = result.strip()
    first_line = stripped.splitlines()[0] if stripped else ""
    return _RUN_FAILED_FALLBACK in first_line


def export_empty(result: str) -> bool:
    """True when an export tool uploaded no images.

    An empty export gives the user nothing to annotate, so it counts as a
    failure even when the export tool itself raised no error.
    """
    return Sentinels.EXPORT_NO_IMAGES in result or bool(_ZERO_IMAGES_RE.search(result))


def parse_run_failure(result: str) -> tuple[str, str]:
    """Return (error line, log path) from a failed run_* tool result."""
    log_match = _LOG_PATH_RE.search(result)
    log_path  = log_match.group(1) if log_match else ""

    err_match = _ERROR_LINE_RE.search(result)
    if err_match:
        return err_match.group(1).strip(), log_path

    # The streaming auto-labeling path returns a fenced stderr block instead
    # of a single "Last error line:" summary. Keep the last real stderr line:
    # the fences and the "failed with exit code N" headline say nothing useful.
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
