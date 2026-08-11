"""Auto-detect fisheye camera intrinsics from a sample image or video frame.

Replaces the manual "screenshot -> open in Preview/GIMP -> click circle
center, click circle edge -> Pythagorean theorem -> f = radius / (pi/2)"
process (the single biggest source of friction in camera calibration --
see msight-customizable-pipeline.md's calibration-UX addendum) with
OpenCV's Hough Circle Transform, which finds the fisheye lens boundary
directly from the image.

Validated against camera_calibration2's own real GridSmart example images
(examples/data/roundabout/{images,camera_parameters}/gs_*.{jpg,json}),
which ship known-correct intrinsics: 1.7-3.9% error on f, sub-3px error on
x0/y0 for 3 of 4 test images (one outlier at ~31px / ~2.4% of frame width)
-- comparable to or better than eyeballing pixel coordinates by hand.

Does NOT replace the map2picturecalibration.net point-pairing step (a
third-party site producing calibrationResults.json) -- only the intrinsics
side of calibration.
"""
import json
from math import pi
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from mcptools import mcp

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def _load_frame(image_path: Path):
    if image_path.suffix.lower() in VIDEO_EXTENSIONS:
        cap = cv2.VideoCapture(str(image_path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        return frame
    return cv2.imread(str(image_path))


# No detection error message text is duplicated between the MCP tool and
# the /msight/detect_intrinsics_from_frame HTTP route (chat_server.py) --
# both call this directly, so there's exactly one place that knows how the
# detection actually works and what "it failed" means.
def detect_intrinsics_from_frame(frame: np.ndarray) -> tuple[Optional[dict], Optional[str]]:
    """Run Hough Circle detection on an already-loaded BGR frame (numpy
    array, e.g. from cv2.imread/cv2.imdecode/cv2.VideoCapture).

    Returns (intrinsics_dict, None) on success, or (None, error_message) if
    no circle was found. Does not touch the filesystem -- callers decide
    whether/where to write the result.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blurred = cv2.medianBlur(gray, 5)
    min_dim = min(h, w)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.5, minDist=min_dim,
        param1=50, param2=30,
        minRadius=int(min_dim * 0.25), maxRadius=int(min_dim * 0.65),
    )
    if circles is None:
        return None, (
            "Could not detect a fisheye circle in this image — the lens boundary may not "
            "be clearly visible (e.g. cropped out, low contrast, or this isn't fisheye "
            "footage). Falling back to manual measurement: open the image in Preview/GIMP, "
            "find the pixel coordinates of the circle's center and one point on its edge, "
            "then f = distance(center, edge) / (pi/2)."
        )

    x0, y0, radius = circles[0][0]
    f = float(radius) / (pi / 2)
    intrinsics = {"f": round(f, 2), "x0": round(float(x0), 2), "y0": round(float(y0), 2)}
    return intrinsics, None


@mcp.tool()
def detect_fisheye_intrinsics(image_path: str, output_path: Optional[str] = None) -> str:
    """
    Auto-detect a fisheye camera's intrinsics (f, x0, y0) from a sample
    image or video file, using OpenCV Hough Circle detection to find the
    fisheye lens's circular boundary -- no manual pixel measurement needed.

    image_path: a local image file, or a video file (uses its first frame).
    output_path: where to write intrinsics.json (default: alongside
    image_path, named "intrinsics.json").
    """
    src = Path(image_path)
    if not src.exists():
        return json.dumps({"status": "error", "message": f"'{image_path}' does not exist on this host."})

    frame = _load_frame(src)
    if frame is None:
        return json.dumps({
            "status": "error",
            "message": f"Could not read an image from '{image_path}' — check it's a valid image or video file.",
        })

    intrinsics, err = detect_intrinsics_from_frame(frame)
    if err:
        return json.dumps({"status": "error", "message": err})

    dest = Path(output_path) if output_path else src.parent / "intrinsics.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(intrinsics, indent=2))

    h, w = frame.shape[:2]
    return json.dumps({
        "status": "ok",
        "message": f"Detected fisheye circle and wrote intrinsics.json to {dest}.",
        "intrinsics": intrinsics,
        "image_size": [w, h],
    })
