"""MCP tool: open a Voxel51 (FiftyOne App) session on a dataset, on port 5151."""
import asyncio
import importlib
import logging
import subprocess
import sys

import config.config as _cc
import fiftyone as fo
import fiftyone.core.odm as _foodm

from msight_agent.shared.host_utils import resolve_host
from msight_agent.shared.mcp import mcp
from msight_agent.shared.paths import ENGINE_DIR, LOG_DIR, SESSION_V51_PATH


def _load(name: str) -> fo.Dataset:
    _foodm.get_db_conn()
    return fo.load_dataset(name)


def _read_config_state() -> dict:
    try:
        importlib.reload(_cc)
        return dict(_cc.WORKFLOW_STATE)
    except Exception as e:
        logging.warning(f"[V51] Error reading WORKFLOW_STATE: {e}")
        return {}


@mcp.tool()
async def launch_voxel51_session(dataset_name: str = "") -> str:
    target_dataset = dataset_name.strip()

    if not target_dataset:
        try:
            state = _read_config_state()
            labeled = state.get("labeled_dataset_name", "")
            base = state.get("dataset_name", "")
            target_dataset = labeled if labeled else base
            logging.warning(f"[V51] Resolved from config state: labeled='{labeled}' base='{base}' -> using '{target_dataset}'")
        except Exception as e:
            logging.warning(f"[V51] Error reading config state: {e}")

    if not target_dataset:
        return "Could not determine which dataset to visualize. Please provide a dataset name."

    # A just-imported dataset can take a few seconds to appear in FiftyOne.
    for attempt in range(6):
        try:
            dataset = await asyncio.to_thread(_load, target_dataset)
            logging.warning(f"[V51] Loaded '{target_dataset}' ({len(dataset)} samples) on attempt {attempt + 1}")
            break
        except Exception as e:
            logging.warning(f"[V51] Load attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(2)
    else:
        return f"Dataset '{target_dataset}' could not be loaded after 10 seconds. Please try again."

    try:
        # Fixed port: kill any existing session so the new one can bind it.
        subprocess.run(
            ["pkill", "-f", "session_v51.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        await asyncio.sleep(1)

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "session_v51.log", "a") as log_file:
            proc = subprocess.Popen(
                [sys.executable, str(SESSION_V51_PATH), target_dataset],
                stdout=log_file,
                stderr=log_file,
                cwd=str(ENGINE_DIR),
            )
        logging.warning(f"[V51] Launched PID={proc.pid} for dataset '{target_dataset}'")
        host = resolve_host()
        return (
            f"Voxel51 session launched for dataset '{target_dataset}'. "
            f"Open your browser and go to: http://{host}:5151"
        )
    except Exception as e:
        logging.exception("[V51] Failed to launch session")
        return f"Failed to launch Voxel51 session: {e}"