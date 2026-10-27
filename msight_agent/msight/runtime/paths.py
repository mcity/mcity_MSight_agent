"""The single place that locates the MSight_Vision checkout (MSIGHT_VISION_PATH)."""
import os
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values

from msight_agent.shared.paths import REPO_ROOT


def get_msight_path(require_compose: bool = True) -> tuple[Optional[Path], Optional[str]]:
    """(path, None) or (None, error). Supervisor-only callers pass require_compose=False.
    .env is re-read on each call so an edit is picked up without a restart."""
    raw = dotenv_values(REPO_ROOT / ".env").get("MSIGHT_VISION_PATH") or os.environ.get("MSIGHT_VISION_PATH")
    if not raw:
        return None, "MSIGHT_VISION_PATH is not set in .env."
    path = Path(raw)
    if not path.is_dir():
        return None, f"MSIGHT_VISION_PATH ('{path}') does not exist or is not a directory."
    if require_compose and not (path / "docker-compose.yml").is_file():
        return None, f"MSIGHT_VISION_PATH ('{path}') does not contain a docker-compose.yml."
    return path, None
