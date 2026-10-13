"""Camera calibration files for the live pipeline: status, labels, reset, and the check tool."""
import hashlib
import subprocess
from pathlib import Path
from typing import Optional

from mcp_layer.mcptools import mcp
from mcp_layer.mcptools.mcp_json import error_json, ok_json
from mcp_layer.mcptools.msight.paths import get_msight_path

# Fixed locations, matching what rfdetr_config.yaml points at.
INTRINSICS_REL = Path("examples/rfdetr/calibration/intrinsics.json")
LOCMAP_REL = Path("examples/rfdetr/locmaps/locmap_sip_gs_Fuller_Glazier2_v1.npz")

# SHA256 of the shipped demo calibration, to tell default from user-uploaded.
_DEFAULT_SHA256 = {
    INTRINSICS_REL: "a04a32ac2bb4e7b54d58769b37cb79c9c7447b4b46c2279db65d05fb2c3eb57a",
    LOCMAP_REL: "367ca8bfc6446efb5b7e1ba3ff5704da66bdcc5d70c35b2415f597c41a2eddd7",
}

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
    intrinsics_hash = _sha256(msight_path / INTRINSICS_REL)
    locmap_hash = _sha256(msight_path / LOCMAP_REL)
    intrinsics_is_default = intrinsics_hash == _DEFAULT_SHA256[INTRINSICS_REL]
    locmap_is_default = locmap_hash == _DEFAULT_SHA256[LOCMAP_REL]

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
    for rel, data in ((INTRINSICS_REL, intrinsics), (LOCMAP_REL, locmap)):
        dest = msight_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)


def reset_calibration_to_default(msight_path: Path) -> None:
    """Restore the demo calibration via `git show HEAD:<path>` (touches only those two files)."""
    def shipped(rel: Path) -> bytes:
        return subprocess.run(["git", "show", f"HEAD:{rel.as_posix()}"],
                              cwd=msight_path, capture_output=True, check=True).stdout

    write_calibration(msight_path, shipped(INTRINSICS_REL), shipped(LOCMAP_REL))


@mcp.tool()
def check_msight_calibration_status() -> str:
    msight_path, err = get_msight_path()
    if err:
        return error_json(err)
    status = calibration_status(msight_path)
    return ok_json(_STATE_TEXT[status["state"]][2], **status)
