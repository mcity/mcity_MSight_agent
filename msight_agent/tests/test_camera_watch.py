"""Camera picture checks (frozen / black / covered / defocused / moved), frame-rate
baselines, and the sentinel's own liveness, on synthetic frames. No camera, no Redis."""
import io
import socket

import msgpack
import numpy as np
import pytest
from PIL import Image

from msight_agent.msight.routes import sentinel_offline_notice
from msight_agent.msight.runtime import incidents
from msight_agent.sentinel.camera_watch import CameraHealth, RateBaseline, decode_frame
from msight_agent.sentinel.watcher import sd_notify

RNG = np.random.default_rng(0)
SCENE = np.clip(np.kron(RNG.uniform(40, 220, (12, 16)), np.ones((10, 10))) + RNG.normal(0, 8, (120, 160)), 0, 255)


def live_frame(t: int) -> np.ndarray:
    """The scene with sensor noise and a 'car' moving across it."""
    frame = SCENE + RNG.normal(0, 3, SCENE.shape)
    x = (t * 9) % 140
    frame[60:75, x:x + 20] = 30
    return frame.astype(np.float32)


def box_blur(frame: np.ndarray, k: int = 9) -> np.ndarray:
    padded = np.pad(frame, k // 2, mode="edge")
    out = np.zeros_like(frame)
    for dy in range(k):
        for dx in range(k):
            out += padded[dy:dy + frame.shape[0], dx:dx + frame.shape[1]]
    return out / (k * k)


FAULTS = {
    "BLACK": lambda t: np.full(SCENE.shape, 16, np.float32),
    "COVERED": lambda t: np.vstack([np.full((102, 160), 16, np.float32), live_frame(t)[102:]]),
    "DEFOCUSED": lambda t: box_blur(live_frame(t)),
    "MOVED": lambda t: np.roll(live_frame(t), (25, 30), axis=(0, 1)),
    "FROZEN": lambda t: live_frame(3) * 0 + FROZEN_FRAME + RNG.normal(0, 0.05, SCENE.shape).astype(np.float32),
}
FROZEN_FRAME = live_frame(3)


def learned_camera() -> CameraHealth:
    camera = CameraHealth(warmup=8, persist=3, clear=2)
    for t in range(8):
        camera.update(live_frame(t))
    assert camera.learned
    return camera


def test_a_normal_camera_raises_nothing():
    camera = learned_camera()
    for t in range(8, 40):
        started, _ = camera.update(live_frame(t))
        assert not started


@pytest.mark.parametrize("condition", sorted(FAULTS))
def test_each_fault_is_reported_after_it_persists_and_clears_after(condition):
    camera = learned_camera()
    seen = [camera.update(FAULTS[condition](t))[0] for t in range(10, 14)]
    assert seen[0] == set() and seen[1] == set()          # one or two bad frames: not yet
    # A freeze shows one sample later: its first frame still differs from the last live one.
    reported_at = 3 if condition == "FROZEN" else 2
    assert seen[reported_at] == {condition}
    cleared = [camera.update(live_frame(t))[1] for t in range(14, 17)]
    assert condition in cleared[0] | cleared[1]


def test_reference_survives_a_restart(tmp_path):
    camera = learned_camera()
    camera.save(tmp_path, "camera_s1")
    restarted = CameraHealth()
    assert restarted.load(tmp_path, "camera_s1")            # knocked while the sentinel was down: still caught
    found = [restarted.update(FAULTS["MOVED"](t))[0] for t in range(4)]
    assert {"MOVED"} in found


def test_decode_frame_reads_msight_image_messages():
    buf = io.BytesIO()
    Image.fromarray(SCENE.astype(np.uint8)).save(buf, format="JPEG")
    payload = msgpack.packb({"data_type": "ImageData", "image": buf.getvalue()}, use_bin_type=True)
    gray = decode_frame(payload)
    assert gray.shape == (120, 160)
    assert decode_frame(b"not msgpack") is None


def test_rate_baseline_flags_a_sustained_drop_only():
    rate = RateBaseline(warmup=4, persist=2)
    for r in (5.0, 5.2, 4.9, 5.1):
        assert rate.update(r) is None
    assert rate.update(1.0) is None        # one low reading
    assert rate.update(1.2) is True        # sustained
    assert rate.update(5.0) is False       # recovered


class FakeRedis:
    def __init__(self, heartbeat: bool, seen):
        self.heartbeat, self.seen = heartbeat, seen

    def exists(self, key):
        return self.heartbeat

    def get(self, key):
        return self.seen


def test_monitor_offline_only_after_a_crash():
    assert incidents.sentinel_offline_since(FakeRedis(heartbeat=True, seen=b"100")) is None
    assert incidents.sentinel_offline_since(FakeRedis(heartbeat=False, seen=b"100")) == 100.0   # crashed
    assert incidents.sentinel_offline_since(FakeRedis(heartbeat=False, seen=None)) is None      # stopped cleanly
    notice = sentinel_offline_notice(100.0)
    assert notice["status"] == "open" and "not detected" in notice["summary"] and incidents.fix_action(notice) is None


def test_sd_notify_speaks_systemds_protocol(tmp_path, monkeypatch):
    path = str(tmp_path / "notify.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(path)
    monkeypatch.setenv("NOTIFY_SOCKET", path)
    sd_notify("WATCHDOG=1")
    assert server.recv(64) == b"WATCHDOG=1"
    server.close()
    monkeypatch.delenv("NOTIFY_SOCKET")
    sd_notify("WATCHDOG=1")                # outside systemd: a no-op


def test_saved_reference_only_applies_to_the_same_source(tmp_path):
    rng = np.random.default_rng(3)
    learned = CameraHealth(warmup=2, source="rtsp://127.0.0.1:8555/cam")
    for _ in range(2):
        learned.update(rng.uniform(0, 255, (120, 160)).astype(np.float32))
    learned.save(tmp_path, "camera_x")

    assert CameraHealth(source="rtsp://127.0.0.1:8555/cam").load(tmp_path, "camera_x")
    assert not CameraHealth(source="/videos/sample.mp4").load(tmp_path, "camera_x")
