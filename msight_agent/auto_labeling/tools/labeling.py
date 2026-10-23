"""MCP tools that configure and run the auto-labeling engine (via config.py and main.py)."""
import asyncio
import logging
import re
import sys
from pathlib import Path

from msight_agent.shared.mcp import mcp
from msight_agent.shared.paths import CONFIG_PATH, LOG_DIR, MAIN_PATH

_RUN_LOG_PATH = LOG_DIR / "last_auto_labeling_log.txt"
_REPORT_HEADER = "              precision    recall  f1-score   support"
_REPORT_END = "You have launched a remote App on port 5151"


def write_run_log(stdout: str, stderr: str, returncode) -> Path:
    """Save a run's full output; returns the log path."""
    _RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RUN_LOG_PATH.write_text(
        f"=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}\n\n=== EXIT CODE ===\n{returncode}",
        encoding="utf-8",
    )
    return _RUN_LOG_PATH


def summarize_run(stdout: str, stderr: str, returncode, log_path: Path) -> str:
    """The run result the agent reads: the classification report (inference), a
    completion note (training), or a RUN_FAILED line with the last error."""
    if returncode != 0:
        logging.warning(f"[AUTO_LABELING] run failed with exit code {returncode}:\n{stderr}")
        tail = stderr.strip().splitlines()[-1] if stderr.strip() else "no error output captured"
        return (
            f"RUN_FAILED: Auto-labeling failed with exit code {returncode}.\n"
            f"Last error line: {tail}\n"
            f"Full logs saved to `{log_path}`"
        )

    error_lines = [line for line in stderr.splitlines() if " - ERROR - " in line]
    if "Evaluating detections..." in stdout + stderr:
        report_lines, capture = [], False
        for line in stdout.splitlines():
            if _REPORT_HEADER in line:
                capture = True
            elif capture and line.startswith(_REPORT_END):
                break
            if capture:
                report_lines.append(line)
        report = "\n".join(report_lines).strip() or "No inference results found."
    elif error_lines:
        report = "Workflow completed with errors:\n" + "\n".join(error_lines[-10:])
    else:
        report = "Workflow completed successfully.\nThe model is ready to be tested using inference on the validation set."
    return (
        f"Auto-labeling workflow completed.\n\n"
        f"**Result Summary:**\n```\n{report}\n```\n"
        f"Full logs saved to `{log_path}`"
    )


def _update_auto_labeling_keys(updated_keys: dict[str, str]) -> bool:
    """Rewrite `"key": value` lines inside WORKFLOWS["auto_labeling"] in config.py.
    Keys are quoted as they appear in the file. False if the block wasn't found."""
    lines = CONFIG_PATH.read_text().split("\n")
    modified = []
    in_block = found = False
    for line in lines:
        if '"auto_labeling": {' in line:
            in_block = found = True
        elif in_block:
            stripped = line.strip()
            key = stripped.split(":")[0]
            if key in updated_keys:
                line = f"        {key}: {updated_keys[key]},"
            elif stripped.startswith("}"):
                in_block = False
        modified.append(line)
    if found:
        CONFIG_PATH.write_text("\n".join(modified).rstrip("\n") + "\n")
    return found



@mcp.tool()
def configure_auto_labeling(selected_source: str, selected_model: str) -> str:
    """
    Enable the selected model source and model inside config.py for the auto_labeling workflow.
    """
    selected_source = selected_source.lower().strip()
    selected_model = selected_model.strip()
    lines = CONFIG_PATH.read_text().split('\n')
    modified = []

    in_model_source = False
    in_selected_source = False
    in_model_dict = False
    in_config_list = False
    in_rf_config_list = False

    source_key_pattern = re.compile(rf'^\s*"{selected_source}"\s*:\s*{{')
    model_line_pattern = re.compile(r'^\s*#?\s*"([^"]+)"\s*:')
    config_line_pattern = re.compile(r'^\s*"([^"]+\.py)"')

    # A line may be commented as '    #"yolo12x"' or '#\t\t"yolo26m"': strip any mix of
    # leading whitespace and '#', then rebuild the indent.
    def _bare(line):
        return re.sub(r'^[\s#]+', '', line).rstrip()

    def _enable(line, indent="                "):
        return indent + _bare(line)

    def _disable(line, indent="                "):
        return indent + "# " + _bare(line)

    enabled_count = 0

    for line in lines:
        stripped = line.strip()

        if '"model_source": [' in line:
            in_model_source = True
            modified.append(line)
            continue
        if in_model_source:
            if ']' in stripped:
                in_model_source = False
                modified.append(line)
                continue
            match = re.search(r'"([^"]+)"', stripped)
            if match:
                source = match.group(1)
                if source == selected_source:
                    modified.append(f'        "{source}",')
                else:
                    modified.append(f'        # "{source}",')
                continue

        if source_key_pattern.match(line):
            in_selected_source = True
            modified.append(line)
            continue

        if in_selected_source:
            if stripped.startswith("}"):
                in_selected_source = False
                modified.append(line)
                continue

            if '"models": {' in line and selected_source == "ultralytics":
                in_model_dict = True
                modified.append(line)
                continue

            if in_model_dict:
                if stripped.startswith("}"):
                    in_model_dict = False
                    modified.append(line)
                    continue
                match = model_line_pattern.match(line)
                if match:
                    model_name = match.group(1)
                    if model_name == selected_model:
                        modified.append(_enable(line))
                        enabled_count += 1
                    else:
                        modified.append(_disable(line))
                    continue

            if selected_source == "hf_models_objectdetection":
                match = model_line_pattern.match(line)
                if match:
                    model_name = match.group(1)
                    if model_name == selected_model:
                        modified.append(_enable(line))
                        enabled_count += 1
                    else:
                        modified.append(_disable(line))
                    continue

            if '"configs": [' in line and selected_source == "custom_codetr":
                in_config_list = True
                modified.append(line)
                continue

            if in_config_list:
                if ']' in stripped:
                    in_config_list = False
                    modified.append(line)
                    continue
                match = config_line_pattern.match(line)
                if match:
                    config_path = match.group(1)
                    if selected_model in config_path:
                        modified.append(_enable(line))
                        enabled_count += 1
                    else:
                        modified.append(_disable(line))
                    continue

            if '"configs": [' in line and selected_source == "roboflow":
                in_rf_config_list = True
                modified.append(line)
                continue

            if in_rf_config_list:
                if ']' in stripped:
                    in_rf_config_list = False
                    modified.append(line)
                    continue

                match = re.search(r'^\s*#?\s*"([^"]+)"', stripped)
                if match:
                    config_name = match.group(1)
                    if config_name == selected_model:
                        indent = len(line) - len(line.lstrip())
                        uncommented = line.lstrip().lstrip('#').lstrip()
                        modified.append(' ' * indent + uncommented)
                        enabled_count += 1
                    else:
                        if not stripped.startswith("#"):
                            indent = len(line) - len(line.lstrip())
                            modified.append(' ' * indent + '# ' + stripped)
                        else:
                            modified.append(line)
                    continue

        modified.append(line)

    # Validate against list_model_sources_and_models()
    valid = list_model_sources_and_models()
    if selected_model not in valid.get(selected_source, []):
        return (
            f"Invalid model '{selected_model}' for source '{selected_source}'. "
            f"Available: {valid.get(selected_source, [])}"
        )
    # Guard against a silent no-op: the model line must really be enabled.
    if enabled_count != 1:
        return (
            f"Config not written: '{selected_model}' matched {enabled_count} lines "
            f"in source '{selected_source}', but exactly 1 is needed. "
            f"Please check the model list in config.py."
        )

    CONFIG_PATH.write_text('\n'.join(modified).rstrip('\n') + '\n')
    return f"Config updated to use `{selected_model}` from `{selected_source}`."


@mcp.tool()
def set_auto_labeling_hyperparams(
    mode: list = None,
    epochs: int = None,
    early_stop_patience: int = None,
    early_stop_threshold: float = None,
    learning_rate: float = None,
    weight_decay: float = None,
    max_grad_norm: float = None,
) -> str:
    """
    Modify only the specified hyperparameters for the auto_labeling workflow.
    """
    values = {
        "mode": mode, "epochs": epochs, "early_stop_patience": early_stop_patience,
        "early_stop_threshold": early_stop_threshold, "learning_rate": learning_rate,
        "weight_decay": weight_decay, "max_grad_norm": max_grad_norm,
    }
    _update_auto_labeling_keys({f'"{k}"': str(v) for k, v in values.items() if v is not None})
    return "Hyperparameters updated successfully."


@mcp.tool()
def set_msight_localization_config(
    detection_field: str,
    enabled: bool = True,
) -> str:
    """
    Configure MSight geolocation for the auto_labeling workflow's detections.
    Writes into config.WORKFLOWS["auto_labeling"], not WORKFLOW_STATE.

    intrinsics_path/locmap_path are NOT settable here -- hardcoded in config.py
    to the Ashley/Huron intersection camera's calibration (single-camera only).
    Only call this tool for a dataset that actually came from that camera.
    """
    updated_keys = {
        "\"localization_enabled\"": str(enabled),
        "\"localization_detection_field\"": repr(detection_field),
    }
    if not _update_auto_labeling_keys(updated_keys):
        return (
            "MSIGHT_LOCALIZATION_ERROR: Could not find the \"auto_labeling\" block in "
            "config.py — localization settings were not saved."
        )

    return (
        "MSight localization configured — it will run automatically as part of the "
        "next auto-labeling run, using the Ashley/Huron intersection camera's "
        "calibration (hardcoded)." if enabled else
        "MSight localization configuration saved (disabled)."
    )


@mcp.tool()
def list_model_sources_and_models() -> dict:
    """
    Lists valid model sources and models for auto_labeling.
    """
    return {
        "ultralytics": ["yolo11n", "yolo11x", "yolo12n", "yolo12x"],
        "hf_models_objectdetection": [
            "microsoft/conditional-detr-resnet-50",
            "Omnifact/conditional-detr-resnet-101-dc5",
            "facebook/detr-resnet-50",
            "facebook/detr-resnet-50-dc5",
            "facebook/detr-resnet-101",
            "facebook/detr-resnet-101-dc5",
            "facebook/deformable-detr-detic",
            "facebook/deformable-detr-box-supervised",
            "SenseTime/deformable-detr",
            "SenseTime/deformable-detr-with-box-refine",
            "jozhang97/deta-swin-large",
            "jozhang97/deta-swin-large-o365",
            "hustvl/yolos-base"
        ],
        "custom_codetr": [
            "co_deformable_detr_r50_1x_coco.py",
            "co_dino_5scale_vit_large_coco.py"
        ],
        "roboflow": ["rfdetr_nano", "rfdetr_small", "rfdetr_medium", "rfdetr_large", "rfdetr_xlarge", "rfdetr_2xlarge"]
    }


@mcp.tool()
async def run_auto_labeling() -> str:
    """Run auto_labeling workflow and return training or inference summary."""
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-u", str(MAIN_PATH),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(MAIN_PATH.parent),
        )

        stdout_data, stderr_data = await process.communicate()
    except Exception as e:
        logging.exception("[AUTO_LABELING] could not start main.py")
        return f"RUN_FAILED: Auto-labeling could not start.\nLast error line: {e}"
    output = stdout_data.decode("utf-8", errors="ignore") if stdout_data else ""
    error_output = stderr_data.decode("utf-8", errors="ignore") if stderr_data else ""
    log_path = write_run_log(output, error_output, process.returncode)
    return summarize_run(output, error_output, process.returncode, log_path)
