"""The source the pipeline was last started on, kept on disk. A source passed to
start_msight_pipeline (an RTSP camera, an uploaded video) isn't in MSight_Vision's .env,
yet restart_msight_node must bring a node back on it, and the sentinel's camera checks
must know when the camera feeding a topic changed. Standard library only, so the
sentinel's venv can import it."""
import json
from typing import Optional

from msight_agent.shared.paths import DATA_DIR

STARTED_SOURCE_PATH = DATA_DIR / "msight" / "started_source.json"


def save(video_input: Optional[str], rtsp_url: Optional[str], sensor: str) -> None:
    try:
        STARTED_SOURCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STARTED_SOURCE_PATH.write_text(json.dumps({"video_input": video_input, "rtsp_url": rtsp_url,
                                                   "sensor": sensor}))
    except OSError:
        pass  # a restart then falls back to MSight_Vision's .env


def load() -> dict:
    """{"video_input", "rtsp_url", "sensor"} as last saved, or {} if never saved or unreadable."""
    try:
        saved = json.loads(STARTED_SOURCE_PATH.read_text())
    except (OSError, ValueError):
        return {}
    return saved if isinstance(saved, dict) else {}


def source() -> Optional[str]:
    """The RTSP URL or video file the pipeline was last started on, if known."""
    saved = load()
    return saved.get("rtsp_url") or saved.get("video_input")
