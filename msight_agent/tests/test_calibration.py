"""Calibration maps must be image-sized: RF-DETR's localizer indexes them by detection pixel."""
import cv2
import numpy as np
import pytest

from msight_agent.msight.runtime import calibration
from msight_agent.msight.runtime.calibration import calibration_mismatch, locmap_problem


@pytest.mark.parametrize("x_shape, y_shape, principal, ok", [
    ((960, 1280), (960, 1280), (640, 480), True),
    ((10, 10), (10, 10), (640, 480), False),          # tiny test map
    ((960, 1280), (960, 1279), (640, 480), False),    # mismatched maps
    ((960, 1280, 3), (960, 1280, 3), (640, 480), False),
])
def test_locmap_problem(x_shape, y_shape, principal, ok):
    assert (locmap_problem(x_shape, y_shape, principal) is None) is ok


def _write_video(path, h, w):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))
    out.write(np.zeros((h, w, 3), np.uint8))
    out.release()


@pytest.fixture
def msight_with_map(tmp_path, monkeypatch):
    def make(h, w):
        config = tmp_path / calibration.RFDETR_CONFIG_REL
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("intrinsics: ./calibration/intrinsics.json\nloc_maps: ./calibration/map.npz\n")
        locmap = config.parent / "calibration" / "map.npz"
        locmap.parent.mkdir(parents=True, exist_ok=True)
        np.savez(locmap, x_map=np.zeros((h, w)), y_map=np.zeros((h, w)))
        return tmp_path
    return make


def test_video_larger_than_map_is_caught(msight_with_map, tmp_path):
    root = msight_with_map(96, 128)
    _write_video(tmp_path / "big.mp4", 104, 128)
    assert "104×128" in calibration_mismatch(root, str(tmp_path / "big.mp4"))


def test_matching_video_passes(msight_with_map, tmp_path):
    root = msight_with_map(96, 128)
    _write_video(tmp_path / "ok.mp4", 96, 128)
    assert calibration_mismatch(root, str(tmp_path / "ok.mp4")) is None


def test_rtsp_and_missing_map_are_skipped(tmp_path):
    assert calibration_mismatch(tmp_path, "rtsp://camera/stream") is None
    _write_video(tmp_path / "v.mp4", 96, 128)
    assert calibration_mismatch(tmp_path, str(tmp_path / "v.mp4")) is None  # no config/map installed


@pytest.mark.parametrize("loc_maps", ["./locmaps/m.npz", "./calibration/m.npz"])
def test_calibration_paths_follow_the_config(tmp_path, loc_maps):
    # Older and newer MSight_Vision layouts both resolve.
    config = tmp_path / calibration.RFDETR_CONFIG_REL
    config.parent.mkdir(parents=True)
    config.write_text(f"intrinsics: ./calibration/intrinsics.json\nloc_maps: {loc_maps}\n")
    intrinsics, locmap = calibration.calibration_paths(tmp_path)
    assert intrinsics == (config.parent / "calibration" / "intrinsics.json").resolve()
    assert locmap == (config.parent / loc_maps).resolve()
