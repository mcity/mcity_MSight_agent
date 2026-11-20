"""A simulated RTSP traffic camera for testing the sentinel's stream monitoring.

    ffmpeg (the camera's encoder) -> mediamtx (the camera, :8554) -> Toxiproxy (:8555) -> video_source

The pipeline reads PIPELINE_URL through Toxiproxy, so network faults are injected without
touching the camera, and picture faults by restarting the encoder with an ffmpeg filter.
mediamtx and Toxiproxy run as Docker containers (named camsim-*, not msight-*, so the
sentinel doesn't mistake them for pipeline nodes); the encoder is a host ffmpeg process.

    python -m msight_agent.sentinel.camera_sim up [--video PATH]   # then start the pipeline on PIPELINE_URL
    python -m msight_agent.sentinel.camera_sim down
    python -m msight_agent.sentinel.camera_sim status

Faults are applied through inject.py (camera-* commands). Stdlib only, so it runs from either venv.
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from msight_agent.shared.paths import LOG_DIR

CAMERA_PORT = 8554
PROXY_PORT = 8555
TOXIPROXY_API = "http://127.0.0.1:8474"
STREAM = "cam"
PIPELINE_URL = f"rtsp://127.0.0.1:{PROXY_PORT}/{STREAM}"
CAMERA_URL = f"rtsp://127.0.0.1:{CAMERA_PORT}/{STREAM}"
MEDIAMTX_IMAGE = "bluenviron/mediamtx:1.9.3"
TOXIPROXY_IMAGE = "ghcr.io/shopify/toxiproxy:2.9.0"
MEDIAMTX = "camsim-mediamtx"
TOXIPROXY = "camsim-toxiproxy"
PROXY_NAME = "camera"
CAMERA_USER, CAMERA_PASS = "camuser", "rotated-password"

STATE_DIR = LOG_DIR / "camera_sim"
PUBLISHER_PID = STATE_DIR / "publisher.pid"
STATE_FILE = STATE_DIR / "state.json"

ZMQ_PORT = 5599  # the encoder's live filter-control socket (ffmpeg's zmq filter)

# Picture modes are applied to the running encoder through ffmpeg's zmq filter, so the
# stream never drops (replacing the encoder would disconnect every reader). The normal view
# is a centred 80% crop, which leaves room to shift it for "moved".
_NORMAL = ["overlay@f x -4000", "avgblur@b sizeX 1", "avgblur@b sizeY 1", "drawbox@k color black@0",
           "overlay@v x -4000", "crop@c x iw*0.1", "crop@c y ih*0.1"]
PICTURES = {
    "normal": [],
    "black": ["drawbox@k color black@1"],
    "blur": ["avgblur@b sizeX 25", "avgblur@b sizeY 25"],                     # defocused lens
    "covered": ["overlay@v x 0"],                                             # top 85% of the lens covered
    "moved": ["crop@c x 0", "crop@c y ih*0.2"],                               # camera knocked: view shifted
    "frozen": ["overlay@f x 0"],                                              # a still frame over the live one
}
_FILTER_GRAPH = (
    f"[0:v]zmq=bind_address=tcp\\\\://127.0.0.1\\\\:{ZMQ_PORT},"
    "crop@c=w=iw*0.8:h=ih*0.8:x=iw*0.1:y=ih*0.1,scale=1280:960,avgblur@b=sizeX=1:sizeY=1,"
    "drawbox@k=x=0:y=0:w=iw:h=ih:color=black@0:t=fill[main];"
    "color=c=black:s=1280x816:r=5[cover];"
    "[main][cover]overlay@v=x=-4000:y=0[covered];"
    "[1:v]crop=iw*0.8:ih*0.8:iw*0.1:ih*0.1,scale=1280:960[still];"
    "[covered][still]overlay@f=x=-4000:y=0:eof_action=repeat[out]"
)


def _mediamtx_config(auth: bool) -> str:
    """TCP only: mediamtx's default RTP/RTCP ports are 8000/8001, which mcp_server and chat_server use."""
    readers = (f"  - user: {CAMERA_USER}\n    pass: {CAMERA_PASS}\n    permissions:\n      - action: read\n"
               if auth else "  - user: any\n    permissions:\n      - action: read\n")
    return ("logLevel: warn\n"
            f"rtspAddress: :{CAMERA_PORT}\nprotocols: [tcp]\nrtpAddress: :18000\nrtcpAddress: :18001\n"
            "rtmp: no\nhls: no\nwebrtc: no\nsrt: no\napi: no\nmetrics: no\n"
            "authInternalUsers:\n"
            "  - user: any\n    ips: ['127.0.0.1']\n    permissions:\n      - action: publish\n"
            f"{readers}"
            f"paths:\n  {STREAM}:\n    source: publisher\n")


def _docker(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(["docker", *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        sys.exit(f"docker {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def _state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _save_state(**changes) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({**_state(), **changes}))


# --- the camera (mediamtx) --------------------------------------------------------------

def start_camera(auth: bool = False) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    config = STATE_DIR / "mediamtx.yml"
    config.write_text(_mediamtx_config(auth))
    _docker("rm", "-f", MEDIAMTX)
    _docker("run", "-d", "--name", MEDIAMTX, "--network", "host",
            "-v", f"{config}:/mediamtx.yml:ro", MEDIAMTX_IMAGE, check=True)
    _save_state(camera="up", auth=auth)


def stop_camera() -> None:
    _docker("stop", "-t", "1", MEDIAMTX)
    _save_state(camera="down")


# --- the network path (Toxiproxy) ------------------------------------------------------

def _toxi(method: str, path: str, body: Optional[dict] = None) -> Optional[dict]:
    request = urllib.request.Request(TOXIPROXY_API + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else None


def start_proxy() -> None:
    _docker("rm", "-f", TOXIPROXY)
    _docker("run", "-d", "--name", TOXIPROXY, "--network", "host", TOXIPROXY_IMAGE, check=True)
    for _ in range(30):
        try:
            _toxi("POST", "/proxies", {"name": PROXY_NAME, "listen": f"127.0.0.1:{PROXY_PORT}",
                                       "upstream": f"127.0.0.1:{CAMERA_PORT}", "enabled": True})
            return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.5)
    sys.exit("Toxiproxy did not come up")


def add_toxic(name: str, toxic_type: str, attributes: dict) -> None:
    """Apply a network fault to the camera's traffic (both directions where it matters)."""
    remove_toxic(name)
    _toxi("POST", f"/proxies/{PROXY_NAME}/toxics",
          {"name": name, "type": toxic_type, "stream": "downstream", "attributes": attributes})


def remove_toxic(name: str) -> None:
    try:
        _toxi("DELETE", f"/proxies/{PROXY_NAME}/toxics/{name}")
    except urllib.error.HTTPError:
        pass


def clear_network() -> None:
    proxy = _toxi("GET", f"/proxies/{PROXY_NAME}") or {}
    for toxic in proxy.get("toxics", []):
        remove_toxic(toxic["name"])
    _toxi("POST", f"/proxies/{PROXY_NAME}", {"enabled": True})


def set_link(enabled: bool) -> None:
    """Disabling the proxy drops every connection and refuses new ones: the camera is unreachable."""
    _toxi("POST", f"/proxies/{PROXY_NAME}", {"enabled": enabled})


# --- the encoder (ffmpeg) ---------------------------------------------------------------

def _still_frame(video: str) -> Path:
    frame = STATE_DIR / "still.png"
    if not frame.is_file():
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", "5", "-i", video,
                        "-frames:v", "1", str(frame)], check=True)
    return frame


def _publisher_cmd(video: str) -> list[str]:
    return ["ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-re", "-stream_loop", "-1", "-i", video,
            "-re", "-loop", "1", "-framerate", "5", "-i", str(_still_frame(video)),
            "-filter_complex", _FILTER_GRAPH, "-map", "[out]", "-an",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-g", "10", "-pix_fmt", "yuv420p",
            "-f", "rtsp", "-rtsp_transport", "tcp", CAMERA_URL]


def _publisher_pid() -> Optional[int]:
    try:
        pid = int(PUBLISHER_PID.read_text())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def start_publisher(video: Optional[str] = None) -> None:
    video = video or _state().get("video")
    if not video:
        sys.exit("No source video: run `camera_sim up --video PATH` first.")
    stop_publisher()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_DIR / "publisher.log", "ab") as log:
        proc = subprocess.Popen(_publisher_cmd(video), stdout=log, stderr=log,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    PUBLISHER_PID.write_text(str(proc.pid))
    _save_state(picture="normal", video=video)
    _wait_for_control_port(proc)


def _wait_for_control_port(proc: subprocess.Popen, timeout: float = 20.0) -> None:
    """The encoder accepts picture commands once its zmq filter is listening."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            sys.exit(f"The camera encoder exited (see {STATE_DIR / 'publisher.log'}).")
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", ZMQ_PORT)) == 0:
                return
        time.sleep(0.25)
    sys.exit(f"The camera encoder didn't open its control port {ZMQ_PORT} within {timeout:.0f}s.")


class EncoderNotAnswering(Exception):
    pass


def _send_commands(commands: list[str]) -> None:
    try:
        import zmq  # in the sentinel venv (pyzmq); the main venv drives this through that venv
    except ImportError:
        sys.exit("Changing the picture needs pyzmq: run camera_sim with .venv-sentinel/bin/python.")
    context = zmq.Context.instance()
    for command in commands:
        sock = context.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, 3000)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://127.0.0.1:{ZMQ_PORT}")
        try:
            sock.send_string(command)
            reply = sock.recv_string()
        except zmq.error.Again:
            raise EncoderNotAnswering(f"The camera encoder didn't answer '{command}'; is the simulator up?")
        finally:
            sock.close()
        if not reply.startswith("0"):
            sys.exit(f"The camera encoder rejected '{command}': {reply}")


def set_picture(picture: str) -> None:
    """Change what the camera sees, live: reset to normal, then apply the mode."""
    if picture not in PICTURES:
        sys.exit(f"Unknown picture mode '{picture}'. Known: {sorted(PICTURES)}")
    if not _publisher_pid():
        start_publisher()
    try:
        _send_commands(_NORMAL + PICTURES[picture])
    except EncoderNotAnswering:
        # ffmpeg can lose its RTSP output (broken pipe) and keep running with its filters
        # stopped: alive, holding the control port, answering nothing. Start it over once.
        start_publisher()
        try:
            _send_commands(_NORMAL + PICTURES[picture])
        except EncoderNotAnswering as e:
            sys.exit(str(e))
    _save_state(picture=picture)


def stop_publisher() -> None:
    """Stop the encoder and wait for it to exit, so a new one can take the control port."""
    if pid := _publisher_pid():
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(0.1)
        else:
            os.kill(pid, signal.SIGKILL)
    PUBLISHER_PID.unlink(missing_ok=True)


# --- lifecycle --------------------------------------------------------------------------

def up(video: str) -> None:
    start_camera(auth=False)
    start_proxy()
    start_publisher(video)
    print(f"Camera simulator up. Start the pipeline on {PIPELINE_URL}")


def down() -> None:
    stop_publisher()
    for name in (TOXIPROXY, MEDIAMTX):
        _docker("rm", "-f", name)
    STATE_FILE.unlink(missing_ok=True)
    print("Camera simulator stopped.")


def restore() -> None:
    """Back to a healthy camera: no auth, normal picture, clean network."""
    state = _state()
    restarted = bool(state.get("auth") or state.get("camera") == "down")
    if restarted:
        start_camera(auth=False)
    clear_network()
    if restarted or not _publisher_pid():
        start_publisher()       # an encoder whose camera went away is dead even if its process isn't
    set_picture("normal")


def status() -> dict:
    running = {name: _docker("inspect", "-f", "{{.State.Running}}", name).stdout.strip() == "true"
               for name in (MEDIAMTX, TOXIPROXY)}
    return {**_state(), "containers": running, "publisher_pid": _publisher_pid(), "pipeline_url": PIPELINE_URL}


NETWORK_FAULTS = {
    # name: (toxiproxy toxic type, attributes) -- applied to the camera -> pipeline direction
    "stall": ("timeout", {"timeout": 0}),            # connection stays open, no data: the reader hangs
    "reset": ("reset_peer", {"timeout": 500}),       # connections are reset shortly after opening
    "slow": ("bandwidth", {"rate": 40}),             # KB/s: frames arrive late, FPS drops
    "latency": ("latency", {"latency": 3000, "jitter": 1000}),
}


def set_network(fault: str) -> None:
    if fault == "clear":
        clear_network()
    elif fault == "offline":
        set_link(False)
    elif fault in NETWORK_FAULTS:
        toxic_type, attributes = NETWORK_FAULTS[fault]
        add_toxic(fault, toxic_type, attributes)
    else:
        sys.exit(f"Unknown network fault '{fault}'. Known: clear, offline, {', '.join(NETWORK_FAULTS)}")
    _save_state(network=fault)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    up_parser = sub.add_parser("up", help="start the camera, proxy and encoder")
    up_parser.add_argument("--video", default=os.environ.get("MSIGHT_DEMO_VIDEO_PATH"))
    sub.add_parser("down", help="stop everything")
    sub.add_parser("status")
    sub.add_parser("restore", help="healthy camera: normal picture, clean network, no auth")
    sub.add_parser("picture", help="change what the camera sees").add_argument("mode", choices=sorted(PICTURES))
    sub.add_parser("network", help="network fault between camera and pipeline").add_argument(
        "fault", choices=["clear", "offline", *NETWORK_FAULTS])
    sub.add_parser("camera", help="power the camera off/on, or reboot it").add_argument(
        "action", choices=["off", "on", "reboot"])
    sub.add_parser("auth", help="camera starts requiring credentials (on) or not (off)").add_argument(
        "state", choices=["on", "off"])
    args = parser.parse_args()

    if args.cmd == "up":
        if not args.video:
            sys.exit("Pass --video or set MSIGHT_DEMO_VIDEO_PATH in .env.")
        up(args.video)
    elif args.cmd == "down":
        down()
    elif args.cmd == "status":
        print(json.dumps(status(), indent=2))
    elif args.cmd == "restore":
        restore()
        print("Camera restored: normal picture, clean network, no auth.")
    elif args.cmd == "picture":
        set_picture(args.mode)
        print(f"Camera picture: {args.mode}")
    elif args.cmd == "network":
        set_network(args.fault)
        print(f"Camera network: {args.fault}")
    elif args.cmd == "camera":
        if args.action in ("off", "reboot"):
            stop_camera()
        if args.action == "reboot":
            time.sleep(8)
        if args.action in ("on", "reboot"):
            start_camera(auth=_state().get("auth", False))
            start_publisher()
        print(f"Camera {args.action}")
    elif args.cmd == "auth":
        start_camera(auth=args.state == "on")
        start_publisher()
        print(f"Camera authentication {args.state}")


if __name__ == "__main__":
    main()
