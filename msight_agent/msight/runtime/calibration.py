"""Camera calibration files for the live pipeline: status, labels, upload, reset."""
import hashlib
import subprocess
from pathlib import Path
from typing import Optional

import yaml

from msight_agent.msight.runtime.paths import get_msight_path

# RF-DETR's config names the calibration files (paths relative to the config).
# Read them from there rather than hard-coding a layout: MSight_Vision has moved
# them before (examples/rfdetr/locmaps/ -> examples/rfdetr/calibration/).
RFDETR_CONFIG_REL = Path("examples/rfdetr/rfdetr_config.yaml")

# SHA256 of the shipped demo calibration, to tell default from user-uploaded.
_DEFAULT_INTRINSICS_SHA256 = "a04a32ac2bb4e7b54d58769b37cb79c9c7447b4b46c2279db65d05fb2c3eb57a"
_DEFAULT_LOCMAP_SHA256 = "367ca8bfc6446efb5b7e1ba3ff5704da66bdcc5d70c35b2415f597c41a2eddd7"


def calibration_paths(msight_path: Path) -> tuple[Path, Path]:
    """(intrinsics, loc_maps) as rfdetr_config.yaml references them."""
    config_path = msight_path / RFDETR_CONFIG_REL
    config = yaml.safe_load(config_path.read_text())
    return (config_path.parent / config["intrinsics"]).resolve(), \
           (config_path.parent / config["loc_maps"]).resolve()

# state -> (SESSION_STATE hint wording, consent-summary wording, check-tool message)
_STATE_TEXT = {
    "missing": (
        "calibration=missing (no calibration files found)",
        "missing (no calibration files found — pipeline may fail to start)",
        "No calibration files found at all.",
    ),
    "default": (
        "calibration=default (demo calibration, no user upload yet)",
        "default demo calibration (no custom calibration uploaded)",
        "Only the shipped demo calibration is present — no user calibration uploaded yet.",
    ),
    "user_calibrated": (
        "calibration=user-uploaded",
        "your uploaded calibration",
        "User-uploaded calibration is active.",
    ),
    "partial": (
        "calibration=partial (inconsistent — one file replaced, one still default)",
        "inconsistent (one file replaced, one still default — re-upload both)",
        "Inconsistent state: one calibration file has been replaced but not the other.",
    ),
}


def _sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def calibration_status(msight_path: Path) -> dict:
    try:
        intrinsics_path, locmap_path = calibration_paths(msight_path)
        intrinsics_hash, locmap_hash = _sha256(intrinsics_path), _sha256(locmap_path)
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        intrinsics_hash = locmap_hash = None  # no readable rfdetr_config.yaml: report as missing
    intrinsics_is_default = intrinsics_hash == _DEFAULT_INTRINSICS_SHA256
    locmap_is_default = locmap_hash == _DEFAULT_LOCMAP_SHA256

    if intrinsics_hash is None or locmap_hash is None:
        state = "missing"
    elif intrinsics_is_default and locmap_is_default:
        state = "default"
    elif not intrinsics_is_default and not locmap_is_default:
        state = "user_calibrated"
    else:
        state = "partial"

    return {
        "state": state,
        "intrinsics_exists": intrinsics_hash is not None,
        "locmap_exists": locmap_hash is not None,
        "intrinsics_is_default": intrinsics_is_default,
        "locmap_is_default": locmap_is_default,
    }


def calibration_message(state: str) -> str:
    return _STATE_TEXT[state][2]


def calibration_label(*, prefixed: bool) -> str:
    """The current calibration state, worded for the SESSION_STATE hint
    (prefixed) or the consent summary (plain)."""
    msight_path, err = get_msight_path()
    if err:
        return ("calibration=unknown (MSIGHT_VISION_PATH not configured)" if prefixed
                else "not available (MSIGHT_VISION_PATH not configured on this host)")
    state = calibration_status(msight_path)["state"]
    hint, summary, _ = _STATE_TEXT[state]
    return hint if prefixed else summary


def write_calibration(msight_path: Path, intrinsics: bytes, locmap: bytes) -> None:
    for dest, data in zip(calibration_paths(msight_path), (intrinsics, locmap)):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)


def reset_calibration_to_default(msight_path: Path) -> None:
    """Restore the demo calibration via `git show HEAD:<path>` (touches only those two files)."""
    def shipped(rel: Path) -> bytes:
        return subprocess.run(["git", "show", f"HEAD:{rel.as_posix()}"],
                              cwd=msight_path, capture_output=True, check=True).stdout

    write_calibration(msight_path, *(shipped(p.relative_to(msight_path.resolve()))
                                     for p in calibration_paths(msight_path)))


def locmap_problem(x_shape: tuple, y_shape: tuple, principal_point: tuple[float, float]) -> Optional[str]:
    """Why a pixel-to-world map can't work, or None. RF-DETR's localizer indexes
    the map directly by detection pixel (row, col), so it must be image-sized."""
    if len(x_shape) != 2 or x_shape != y_shape:
        return f"x_map and y_map must be 2-D arrays of the same shape (got {x_shape} and {y_shape})."
    h, w = x_shape
    x0, y0 = principal_point
    if not (0 <= x0 < w and 0 <= y0 < h):
        return (f"the maps are {h}×{w}, but the camera's principal point (x0={x0}, y0={y0}) lies outside "
                "them -- the maps must have one entry per image pixel (height × width of the camera image).")
    return None


def _video_frame_size(video_input: str) -> Optional[tuple[int, int]]:
    """(height, width) of a video file, or of the first .mp4 in a folder; None if unreadable."""
    import cv2
    path = Path(video_input)
    if path.is_dir():
        path = next(iter(sorted(path.glob("*.mp4"))), None)
        if path is None:
            return None
    cap = cv2.VideoCapture(str(path))
    try:
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    finally:
        cap.release()
    return (h, w) if h and w else None


def calibration_mismatch(msight_path: Path, video_input: Optional[str]) -> Optional[str]:
    """Why the active calibration map can't localize this video's frames, or None.
    A map smaller than the frame crashes RF-DETR on its first detection outside it.
    RTSP sources can't be checked cheaply up front, so they're skipped."""
    import numpy as np
    if not video_input or not Path(video_input).exists():
        return None
    try:
        map_h, map_w = np.load(calibration_paths(msight_path)[1])["x_map"].shape[:2]
    except Exception:
        return None  # missing/unreadable calibration is reported by calibration_status
    frame = _video_frame_size(video_input)
    if frame is None or (frame[0] <= map_h and frame[1] <= map_w):
        return None
    return (f"The active calibration map is {map_h}×{map_w}, but this video's frames are "
            f"{frame[0]}×{frame[1]} -- RF-DETR would crash on the first detection outside the map. "
            "Upload a calibration built for this camera's resolution, or use a video that matches it.")
