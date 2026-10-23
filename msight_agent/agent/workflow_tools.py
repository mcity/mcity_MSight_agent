import asyncio
import logging
import re
import time
from pathlib import Path
from typing import List
from unittest import mock

import fiftyone as fo
import fiftyone.core.odm as _foodm
from fastmcp import Context
from ruamel.yaml import YAML
from tqdm import tqdm as _tqdm_base

from utils.dataset_loader import load_dataset
from msight_agent.agent.state import WorkflowState
from msight_agent.shared.mcp import mcp
from msight_agent.shared.paths import CONFIG_PATH, DATASETS_YAML as DEFAULT_DATASETS_YAML

# One line only ([ \t], not \s): \s would also swallow the blank lines around it.
SELECTED_WORKFLOW_RE = re.compile(r"^[ \t]*SELECTED_WORKFLOW[ \t]*=[ \t]*\[[^\n]*\][ \t]*$", re.MULTILINE)


def with_selected_workflow(src: str, workflow_name: str) -> str:
    """config.py source with SELECTED_WORKFLOW set to [workflow_name] (prepended if absent)."""
    line = f'SELECTED_WORKFLOW = ["{workflow_name}"]'
    if SELECTED_WORKFLOW_RE.search(src):
        return SELECTED_WORKFLOW_RE.sub(line, src, count=1)
    return f"{line}\n{src}"


@mcp.tool()
def select_workflow(workflow_name: str) -> str:
    """Updates SELECTED_WORKFLOW in config.py to the given workflow name."""
    CONFIG_PATH.write_text(with_selected_workflow(CONFIG_PATH.read_text(), workflow_name))
    return f"Workflow selected: `{workflow_name}`."


def _prune_stale_custom_entries() -> None:
    """Drop datasets.yaml entries for custom datasets no longer in FiftyOne."""
    try:
        _foodm.get_db_conn()
        existing = set(fo.list_datasets())
    except Exception as e:
        logging.warning(f"[DATASETS] Could not verify against FiftyOne, skipping prune: {e}")
        return

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(sequence=4, offset=2)
    try:
        with open(DEFAULT_DATASETS_YAML, "r") as f:
            data = yaml.load(f)
        entries = data.get("datasets", []) or []
        stale = [
            d for d in entries
            if str(d.get("loader_fct", "")) == "load_custom_dataset" and str(d.get("name", "")) not in existing
        ]
        if not stale:
            return
        data["datasets"] = [d for d in entries if d not in stale]
        with open(DEFAULT_DATASETS_YAML, "w") as f:
            yaml.dump(data, f)
        logging.warning(f"[DATASETS] Pruned stale entries no longer in FiftyOne: {[str(d.get('name')) for d in stale]}")
    except Exception as e:
        logging.warning(f"[DATASETS] Failed to prune stale entries: {e}")


def write_selected_dataset(dataset_name: str) -> None:
    """Point SELECTED_DATASET in config.py (what main.py runs on) at `dataset_name`."""
    lines = CONFIG_PATH.read_text().split("\n")
    modified = []
    in_dataset_block = False

    for line in lines:
        if "SELECTED_DATASET = {" in line:
            in_dataset_block = True
            modified.append(line)
            continue
        if in_dataset_block:
            if '"name":' in line:
                indent = " " * (len(line) - len(line.lstrip()))
                modified.append(f'{indent}"name": "{dataset_name}",')
                continue
            elif '"n_samples":' in line:
                indent = " " * (len(line) - len(line.lstrip()))
                modified.append(f'{indent}"n_samples": None,')
                continue
            elif '"custom_view":' in line:
                indent = " " * (len(line) - len(line.lstrip()))
                modified.append(f'{indent}"custom_view": None,')
                continue
            elif "}" in line:
                modified.append(line)
                in_dataset_block = False
                continue
        modified.append(line)

    CONFIG_PATH.write_text("\n".join(modified).rstrip("\n") + "\n")


async def _load_dataset_with_progress(dataset_name: str, ctx) -> None:
    """load_dataset() in a worker thread, relaying ~30s download progress over the MCP
    log channel. FiftyOne has no progress callback; its HF loader imports tqdm per call,
    so patching tqdm.tqdm is the only hook that takes effect."""
    loop = asyncio.get_running_loop()
    last_emit = 0.0

    class _ProgressTqdm(_tqdm_base):
        def update(self, n=1):
            nonlocal last_emit
            result = super().update(n)
            if ctx and self.total:
                now = time.time()
                is_done = self.n >= self.total
                if now - last_emit >= 30 or is_done:
                    last_emit = now
                    pct = self.n / self.total * 100
                    msg = f"Downloading {dataset_name}: {self.n}/{self.total} ({pct:.0f}%)"
                    rate = self.format_dict.get("rate")
                    if rate and not is_done:
                        remaining = (self.total - self.n) / rate
                        msg += f", ~{_tqdm_base.format_interval(remaining)} remaining"
                    asyncio.run_coroutine_threadsafe(ctx.log(msg), loop)
            return result

    with mock.patch("tqdm.tqdm", _ProgressTqdm):
        await asyncio.to_thread(
            load_dataset,
            {"name": dataset_name, "n_samples": None, "custom_view": None},
        )


@mcp.tool()
async def set_selected_dataset(dataset_name: str, ctx: Context = None) -> str:
    """
    Updates SELECTED_DATASET section in config.py with the given dataset name.
    Always uses the full dataset (n_samples = None).
    """
    _prune_stale_custom_entries()
    _foodm.get_db_conn()
    available = fo.list_datasets()
    if dataset_name not in available:
        yaml_names = _extract_names_after_line(DEFAULT_DATASETS_YAML, 0)
        all_known = set(available + yaml_names + FIXED_DATASETS)
        if dataset_name not in all_known:
            return f"DATASET_NOT_FOUND: '{dataset_name}' does not exist."

    write_selected_dataset(dataset_name)

    if not dataset_name.startswith("custom"):
        await _load_dataset_with_progress(dataset_name, ctx)

    return f"Dataset set to `{dataset_name}`."


def _reset_selected_dataset_block(lines: list[str]) -> list[str]:
    """Return `lines` with the SELECTED_DATASET block reset to empty defaults."""
    result_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "SELECTED_DATASET = {" in line:
            result_lines.append("SELECTED_DATASET = {")
            result_lines.append('    "name": "",')
            result_lines.append('    "n_samples": None,')
            result_lines.append('    "custom_view": None,')
            result_lines.append("}")
            while i < len(lines) and lines[i].strip() != "}":
                i += 1
        else:
            result_lines.append(line)
        i += 1
    return result_lines


@mcp.tool()
def switch_workflow(workflow_name: str) -> str:
    """
    Switch to a new workflow. Updates SELECTED_WORKFLOW and clears SELECTED_DATASET
    in config.py, since every caller treats this as a full reset.
    """
    lines = with_selected_workflow(CONFIG_PATH.read_text(), workflow_name).split("\n")
    CONFIG_PATH.write_text("\n".join(_reset_selected_dataset_block(lines)).rstrip("\n") + "\n")
    return f"Switched to workflow: `{workflow_name}`."


@mcp.tool()
def reset_workflow_state() -> str:
    """
    Resets SELECTED_WORKFLOW, SELECTED_DATASET, and WORKFLOW_STATE in config.py,
    allowing the user to start a new workflow.
    """
    WorkflowState.reset()

    lines = with_selected_workflow(CONFIG_PATH.read_text(), "").split("\n")
    CONFIG_PATH.write_text("\n".join(_reset_selected_dataset_block(lines)).rstrip("\n") + "\n")
    return "Workflow, dataset, and session state have been reset. You may now start a new workflow."


# Dataset listing

FIXED_DATASETS: List[str] = [
    "fisheye8k",
    "fisheye8k_mini",
    "mcity_fisheye_2000",
    "mcity_fisheye_2100",
]

# list_datasets offers the fixed datasets plus the datasets.yaml entries from this line
# on (the entries above it are engine-internal datasets the chat doesn't offer).
_FIRST_LISTED_YAML_LINE = 52

_NAME_LINE = re.compile(r"^\s*-\s*name:\s*[\"']?([^\"']+)[\"']?\s*$", re.IGNORECASE)


def _extract_names_after_line(yaml_path: Path, start_line_1_based: int) -> List[str]:
    if not yaml_path.exists():
        return []
    lines = yaml_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    tail = lines[start_line_1_based:]
    found: List[str] = []
    for raw in tail:
        m = _NAME_LINE.match(raw)
        if not m:
            continue
        name = m.group(1).strip()
        if name and name not in found:
            found.append(name)
    return found


@mcp.tool()
def list_datasets() -> List[str]:
    """
    Returns all available dataset names.
    Fixed datasets always appear first, followed by names from datasets.yaml.
    """
    _prune_stale_custom_entries()
    dynamic_names = _extract_names_after_line(DEFAULT_DATASETS_YAML, _FIRST_LISTED_YAML_LINE)
    out: List[str] = []
    for n in FIXED_DATASETS:
        if n not in out:
            out.append(n)
    for n in dynamic_names:
        if n not in out:
            out.append(n)
    return out