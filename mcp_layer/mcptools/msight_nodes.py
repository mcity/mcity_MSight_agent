"""Generic node primitives, exposed as real tools -- the "build your own
topology" surface, distinct from msight_docker.py's fixed 3-node pipeline
and msight_record_archive.py's fixed record/archive chain.

These are thin delegations to MSightControlPlane (already built, already
live-tested) -- deliberately not gated by any consent mechanism (see the
plan doc, Workstream D): add/remove of a single node is cheap, fast, and
reversible, exactly like the existing ungated stop_msight_pipeline/
stop_msight_recording/stop_msight_archiving tools.
"""
from typing import Optional

from mcptools import mcp
from mcptools.mcp_json import error_json, ok_json
from mcptools.msight_node_catalog import NODE_CATALOG
from mcptools.msight_docker import _control_plane, _get_msight_path

_NODE_TYPE_DESCRIPTIONS: dict[str, str] = {
    "video_source_rtsp": "Publishes frames from a live RTSP stream (or a local video file path passed as rtsp_url).",
    "video_source_mp4_folder": "Publishes frames sequentially from a folder of .mp4 files.",
    "rfdetr_detector": "Runs RF-DETR object detection on a frame topic, publishes detections. Needs GPU.",
    "detection_viewer": "Web viewer (configurable port) that renders a detection topic as annotated video in-browser.",
    "frame_annotator": "Draws detection boxes/labels onto frames, republishes as an annotated frame topic.",
    "video_aggregator": "Buffers an annotated frame topic into short video clips.",
    "video_local_dumper": "Writes a video topic's clips to local disk.",
    "aws_video_pusher": "Pushes a video topic's clips to an S3 bucket.",
    "aggregator": "Generic frame-buffering aggregator (distinct from video_aggregator's image-to-video specialization).",
    "aws_sequence_pusher": "Pushes a raw data sequence topic to an S3 bucket.",
    "buffering_sort": "Buffers and time-sorts an out-of-order topic.",
    "bytes_viewer": "Logs raw bytes-topic messages -- a plain text logger, no GUI.",
    "detection_results_viewer": "Logs detection results as text -- no GUI, safe for headless hosts (see 2d_viewer for a GUI equivalent).",
    "http_sink": "POSTs a topic's messages to an external HTTP endpoint.",
    "ifm": "Pushes data to a roadside unit (RSU) over IFM. Needs a real, reachable RSU device.",
    "image_viewer": "Renders an image topic in a GUI window (cv2.imshow) -- needs a real display/X11, not headless-safe.",
    "local_image": "Replays a single local image file as a source topic, at a configurable fps -- simple test/dummy source.",
    "pointcloud_local_dumper": "Writes a point-cloud topic's data to local disk.",
    "pointcloud_viewer": "Renders a point-cloud topic in a 3D GUI window (open3d) -- needs a real display, not headless-safe.",
    "sdsm_encoder": "Encodes detections into SDSM (Sensor Data Sharing Message) format.",
    "udp_server": "Listens for incoming UDP packets on a port, publishes them as a source topic -- needs an external sender.",
    "velodyne_lidar": "Ingests a real Velodyne LiDAR unit's UDP stream as a source topic -- needs real hardware (or a UDP replay of one).",
    "websocket_client": "Connects to an external WebSocket server, publishes received messages as a source topic.",
    "custom_fuser": "Multi-sensor fusion node, driven by a YAML fusion config file. sensor_name is required (the underlying node hard-asserts it, unlike other processing nodes).",
    "finite_difference_state_estimator": "Estimates object velocity/heading via finite differences, driven by a YAML estimator config file.",
    "sort_tracker": "SORT multi-object tracker, driven by a YAML tracking config file.",
    "yolo_onestage_detection": "Runs YOLO object detection on a frame topic. Uses GPU automatically if present (via the local venv, not a fixed image -- no CPU/GPU image swap needed, unlike rfdetr_detector).",
    "2d_viewer": "Renders detection results in a GUI window (cv2.imshow) -- needs a real display/X11, not headless-safe.",
    "road_user_list_viewer": "Renders a road-user list on a basemap image in a GUI window -- needs a real display/X11 and a basemap image file, not headless-safe.",
}


@mcp.tool()
def list_msight_node_types() -> str:
    catalog = [
        {
            "node_type": node_type,
            "category": spec.category.value,
            "description": _NODE_TYPE_DESCRIPTIONS.get(node_type, ""),
            "required_config": list(spec.required_config),
            "needs_gpu": spec.needs_gpu,
            "default_name": spec.default_name,
        }
        for node_type, spec in sorted(NODE_CATALOG.items())
    ]
    return ok_json(node_types=catalog)


@mcp.tool()
async def add_msight_node(node_type: str, name: Optional[str] = None, config: Optional[dict] = None) -> str:
    msight_path, err = _get_msight_path()
    if err:
        return error_json(err)
    if node_type not in NODE_CATALOG:
        return error_json(f"Unknown node_type '{node_type}'. Call list_msight_node_types() first.")

    spec = NODE_CATALOG[node_type]
    resolved_name = name or spec.default_name or node_type
    cp = _control_plane()

    # Omitting `name` collides with the catalog default (e.g. a second
    # detection_viewer reusing "detection_viewer"), silently reusing the
    # existing node instead of starting a new one -- surface that plainly.
    # is_alive(), not is_tracked(): add_node() itself only reuses an entry
    # that's verified alive, so this flag has to agree with that.
    already_running = await cp.is_alive(resolved_name)

    try:
        await cp.add_node(node_type, name=name, config=config)
    except Exception as e:
        return error_json(f"Could not start '{resolved_name}' ({node_type}): {e}")

    if already_running:
        return ok_json(
            f"'{resolved_name}' was already running -- reused as-is, no new node was "
            f"started (add_msight_node is idempotent by name). If you wanted a SEPARATE "
            f"node, call again with a different `name`.",
            name=resolved_name, node_type=node_type, already_running=True,
        )
    return ok_json(
        f"'{resolved_name}' ({node_type}) started.",
        name=resolved_name, node_type=node_type, already_running=False,
    )


@mcp.tool()
async def remove_msight_node(name: str) -> str:
    msight_path, err = _get_msight_path()
    if err:
        return error_json(err)

    cp = _control_plane()
    was_tracked = cp.is_tracked(name)
    try:
        await cp.delete_node(name)
    except Exception as e:
        return error_json(f"Could not stop '{name}': {e}")

    msg = f"'{name}' stopped." if was_tracked else f"'{name}' was not tracked; nothing to stop."
    return ok_json(msg, name=name)
