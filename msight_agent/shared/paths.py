"""Every repo path the agent uses, resolved from the repo root (never the working directory).

The auto-labeling engine (repo-root auto_labeling/) is a separate program: it runs as
`python main.py` from its own folder and imports its modules as top-level packages
(config, utils, workflows). The agent reads its config and dataset loader the same
way, so the engine folder is put on sys.path here, once, when msight_agent is imported.

Everything the agent writes at runtime goes under data/ (gitignored).
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- auto-labeling engine ---
ENGINE_DIR = REPO_ROOT / "auto_labeling"
CONFIG_PATH = ENGINE_DIR / "config" / "config.py"
DATASETS_YAML = ENGINE_DIR / "config" / "datasets.yaml"
MAIN_PATH = ENGINE_DIR / "main.py"
SESSION_V51_PATH = ENGINE_DIR / "session_v51.py"

if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

# --- runtime data ---
DATA_DIR = REPO_ROOT / "data"
LOG_DIR = DATA_DIR / "logs"
NODE_LOG_DIR = LOG_DIR / "msight_nodes"  # host-process node logs + pidfiles
MSIGHT_RECORDINGS_DIR = Path(os.environ.get("MSIGHT_RECORDING_SAVE_DIR") or DATA_DIR / "msight" / "recordings")
MSIGHT_DOWNLOADS_DIR = DATA_DIR / "msight" / "downloads"
MSIGHT_UPLOADS_DIR = DATA_DIR / "msight" / "uploads"
DATASET_UPLOADS_DIR = Path(os.environ.get("MDE_UPLOAD_ROOT") or DATA_DIR / "dataset_uploads")
ANNOTATION_TASKS_DIR = DATA_DIR / "annotation_tasks"
