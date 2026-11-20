"""Checks on the camera picture itself, for faults where every process keeps running but
what the pipeline sees is wrong (the "silent" camera faults of the camera-tampering
literature) plus frame-rate drops:

  BLACK        the image is (almost) black
  COVERED      most of the lens is blocked: much darker than normal
  DEFOCUSED    much less sharp than normal (fog, smear, focus)
  FROZEN       consecutive frames are identical: the stream repeats one image
  MOVED        the view no longer matches the camera's reference view
  LOW_FPS      the camera delivers far fewer frames than normal
  SLOW_DETECTOR  frames arrive normally, but detections come out far slower

Each sample is one frame from the camera topic (~75 KB), reduced to a small grayscale
image. Normal levels are learned per camera during a warm-up and saved, so a camera that
was knocked while the sentinel was down is still caught. A condition is reported only
after it persists, and cleared after it has been gone for a while.
"""
import io
import json
import time
from pathlib import Path
from typing import Optional

import msgpack
import numpy as np
from PIL import Image

SAMPLE_SECONDS = 5
FPS_EVERY_SECONDS = 30
FPS_WINDOW_SECONDS = 2.0
WARMUP_SAMPLES = 8
PERSIST_SAMPLES = 3       # ~15 s before a condition is reported
CLEAR_SAMPLES = 2
SMALL = (160, 120)

DESCRIPTIONS = {
    "BLACK": "the camera image is black",
    "COVERED": "most of the camera's view is blocked (the image is far darker than normal)",
    "DEFOCUSED": "the camera image is blurred (far less sharp than normal)",
    "FROZEN": "the camera keeps sending the same image (the picture is frozen)",
    "MOVED": "the camera's view no longer matches its reference view (it may have been moved)",
    "LOW_FPS": "the camera is delivering far fewer frames than normal",
    "SLOW_DETECTOR": "frames arrive normally but detections come out far slower than normal",
}


# --- frames and measurements (pure) -------------------------------------------------------

def decode_frame(payload: bytes) -> Optional[np.ndarray]:
    """A small float grayscale image from one msight_core ImageData message, or None."""
    try:
        message = msgpack.unpackb(payload, raw=False)
        image = Image.open(io.BytesIO(message["image"])).convert("L").resize(SMALL)
    except Exception:  # noqa: BLE001 -- not an image message
        return None
    return np.asarray(image, dtype=np.float32)


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: high for crisp edges, near zero for a blurred or flat image."""
    lap = (gray[1:-1, :-2] + gray[1:-1, 2:] + gray[:-2, 1:-1] + gray[2:, 1:-1] - 4 * gray[1:-1, 1:-1])
    return float(lap.var())


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation of two images, -1..1 (1 = same view)."""
    a, b = a - a.mean(), b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom else 0.0


def measure(gray: np.ndarray, previous: Optional[np.ndarray], reference: Optional[np.ndarray]) -> dict:
    return {
        "brightness": round(float(gray.mean()), 1),
        "sharpness": round(sharpness(gray), 1),
        "change": round(float(np.abs(gray - previous).mean()), 2) if previous is not None else None,
        "similarity": round(similarity(gray, reference), 3) if reference is not None else None,
    }


class CameraHealth:
    """One camera's learned normal picture and current conditions."""

    def __init__(self, warmup: int = WARMUP_SAMPLES, persist: int = PERSIST_SAMPLES, clear: int = CLEAR_SAMPLES,
                 source: Optional[str] = None):
        self.warmup, self.persist, self.clear = warmup, persist, clear
        self.source = source              # what feeds this camera topic (an RTSP URL or a video file)
        self.learning: list[np.ndarray] = []
        self.reference: Optional[np.ndarray] = None
        self.baseline: dict = {}
        self.previous: Optional[np.ndarray] = None
        self.streaks: dict[str, int] = {}
        self.active: set[str] = set()
        self.last: dict = {}

    @property
    def learned(self) -> bool:
        return self.reference is not None

    def _learn(self, gray: np.ndarray) -> None:
        self.learning.append(gray)
        if len(self.learning) < self.warmup:
            return
        stack = np.stack(self.learning)
        self.reference = np.median(stack, axis=0)        # background: passing traffic averages out
        changes = [float(np.abs(stack[i] - stack[i - 1]).mean()) for i in range(1, len(stack))]
        sims = [similarity(g, self.reference) for g in stack]
        self.baseline = {"brightness": float(stack.mean()), "sharpness": float(np.median([sharpness(g) for g in stack])),
                         "change": float(np.median(changes)), "similarity": float(np.median(sims)),
                         "similarity_spread": float(np.std(sims))}
        self.learning = []

    def conditions(self, m: dict) -> set[str]:
        """What is wrong in one measured frame (most specific cause only)."""
        b = self.baseline
        # Video "black" is ~16, not 0 (limited-range YUV); black and flat means no picture at all.
        if m["brightness"] < 30 and m["sharpness"] < 5:
            return {"BLACK"}
        if m["brightness"] < 0.4 * b["brightness"]:
            return {"COVERED"}
        if m["sharpness"] < 0.2 * b["sharpness"]:
            return {"DEFOCUSED"}
        found = set()
        if m["similarity"] is not None and m["similarity"] < b["similarity"] - max(0.2, 6 * b["similarity_spread"]):
            found.add("MOVED")
        # Re-encoding one repeated image still leaves ~0.2 of JPEG noise; a live scene changes
        # several times more. (A static night scene could approach this: it must persist.)
        if m["change"] is not None and b["change"] > 0.5 and m["change"] < max(0.5, 0.15 * b["change"]):
            found.add("FROZEN")
        return found

    def update(self, gray: np.ndarray) -> tuple[set[str], set[str]]:
        """Feed one frame; returns (conditions that just started, conditions that just cleared)."""
        if not self.learned:
            self._learn(gray)
            self.previous = gray
            return set(), set()
        self.last = measure(gray, self.previous, self.reference)
        self.previous = gray
        now = self.conditions(self.last)
        started, cleared = set(), set()
        for name in set(self.streaks) | now:
            if name in now:
                self.streaks[name] = max(1, self.streaks.get(name, 0) + 1)
                if self.streaks[name] >= self.persist and name not in self.active:
                    self.active.add(name)
                    started.add(name)
            else:
                self.streaks[name] = min(0, self.streaks.get(name, 0)) - 1
                if -self.streaks[name] >= self.clear and name in self.active:
                    self.active.discard(name)
                    cleared.add(name)
        return started, cleared

    # persistence: a camera knocked while the sentinel was down is still caught. The saved
    # view belongs to one source: a topic fed from a different camera or video relearns.
    def save(self, directory: Path, key: str) -> None:
        if not self.learned:
            return
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / f"{key}.npy", self.reference)
        (directory / f"{key}.json").write_text(json.dumps(
            {"baseline": self.baseline, "source": self.source, "saved_at": time.time()}))

    def load(self, directory: Path, key: str) -> bool:
        try:
            reference = np.load(directory / f"{key}.npy")
            saved = json.loads((directory / f"{key}.json").read_text())
            baseline = saved["baseline"]
        except (OSError, ValueError, KeyError):
            return False
        if reference.shape != (SMALL[1], SMALL[0]):
            return False
        if saved.get("source") != self.source:
            return False
        self.reference, self.baseline = reference, baseline
        return True


class RateBaseline:
    """A topic's normal message rate, and whether it has dropped below half of it."""

    def __init__(self, warmup: int = 4, persist: int = 2, factor: float = 0.5):
        self.warmup, self.persist, self.factor = warmup, persist, factor
        self.samples: list[float] = []
        self.normal: Optional[float] = None
        self.low_streak = 0
        self.low = False

    def update(self, rate: float) -> Optional[bool]:
        """True when it just became low, False when it just recovered, None otherwise."""
        if self.normal is None:
            self.samples.append(rate)
            if len(self.samples) >= self.warmup:
                self.normal = float(np.median(self.samples))
            return None
        if rate < self.factor * self.normal:
            self.low_streak += 1
            if self.low_streak >= self.persist and not self.low:
                self.low = True
                return True
        else:
            self.low_streak = 0
            if self.low:
                self.low = False
                return False
        return None


def save_frame(payload: bytes, path: Path) -> Optional[str]:
    """Write the frame's JPEG to `path` (for the report and for Claude to look at)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(msgpack.unpackb(payload, raw=False)["image"])
        return str(path)
    except Exception:  # noqa: BLE001
        return None


def save_reference(reference: np.ndarray, path: Path) -> Optional[str]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(reference.astype(np.uint8)).save(path)
        return str(path)
    except Exception:  # noqa: BLE001
        return None
