"""Data-driven catalog of msight_core node types this agent can run.

resolve_cmd() adds the shared base flags (--name, --publish-topic,
--subscribe-topic, --sensor-name per category), so a NodeSpec only declares
what's node-specific. To add a node type, add a NodeSpec after checking its
real flags in MSight_Vision/cli/launch_*.py.
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class NodeCategory(str, Enum):
    SOURCE = "source"
    PROCESSING = "processing"
    SINK = "sink"


@dataclass(frozen=True)
class NodeSpec:
    node_type: str
    category: NodeCategory
    binary: str
    invocation: str  # "docker" | "supervisor" -- which executor backend runs this node type
    required_config: tuple[str, ...] = ()
    arg_map: dict[str, str] = field(default_factory=dict)
    positional_flags: tuple[str, ...] = ()
    needs_gpu: bool = False
    default_name: str | None = None
    # Docker only: if config[mount_config_key] is a host path, mount it ro and rewrite the flag.
    mount_config_key: str | None = None
    mount_container_path: str | None = None
    # Supervisor only: run `python <script_path>` for the agent's own node scripts.
    script_path: Path | None = None
    # Always-on mounts: (path relative to MSIGHT_VISION_PATH, container path, "ro"|"rw").
    fixed_mounts: tuple[tuple[str, str, str], ...] = ()


NODE_CATALOG: dict[str, NodeSpec] = {
    # --- docker-compose.yml's fixed stack (docker-executed) ---
    "video_source_rtsp": NodeSpec(
        node_type="video_source_rtsp",
        category=NodeCategory.SOURCE,
        binary="msight_launch_rtsp",
        invocation="docker",
        required_config=("rtsp_url",),
        arg_map={"rtsp_url": "--url", "rtsp_transport": "--rtsp-transport"},
        default_name="video_source",
        # --url also accepts a local file; mount it when it isn't a URL.
        mount_config_key="rtsp_url",
        mount_container_path="/input",
    ),
    "video_source_mp4_folder": NodeSpec(
        node_type="video_source_mp4_folder",
        category=NodeCategory.SOURCE,
        binary="msight_launch_mp4_folder",
        invocation="docker",
        required_config=("folder",),
        arg_map={"folder": "--folder", "resize_ratio": "--resize-ratio"},
        default_name="video_source",
        mount_config_key="folder",
        mount_container_path="/input",
    ),
    "rfdetr_detector": NodeSpec(
        node_type="rfdetr_detector",
        category=NodeCategory.PROCESSING,
        binary="msight_launch_rfdetr_detection",
        invocation="docker",
        required_config=("det_configs",),
        arg_map={"det_configs": "--det-configs"},
        needs_gpu=True,
        default_name="rfdetr_detector",
        fixed_mounts=(
            ("examples/rfdetr/rfdetr_config.yaml", "/configs/rfdetr_config.yaml", "ro"),
            ("examples/rfdetr/calibration", "/configs/calibration", "ro"),
            ("examples/rfdetr/locmaps", "/configs/locmaps", "ro"),
            ("examples/rfdetr/models", "/configs/models", "rw"),
        ),
    ),
    "detection_viewer": NodeSpec(
        node_type="detection_viewer",
        category=NodeCategory.SINK,
        binary="msight_launch_web_viewer",
        invocation="docker",
        required_config=(),
        arg_map={"port": "--port"},
        default_name="detection_viewer",
    ),
    # --- msight_record_archive.py's fixed subprocess chain (supervisor-executed) ---
    "frame_annotator": NodeSpec(
        node_type="frame_annotator",
        category=NodeCategory.PROCESSING,
        binary="python3",
        invocation="supervisor",
        default_name="frame_annotator",
        # Our own script; it has no --sensor-name flag, so never pass sensor_name.
        script_path=Path(__file__).resolve().parents[1] / "msight_nodes" / "annotated_frame_publisher.py",
    ),
    "video_aggregator": NodeSpec(
        node_type="video_aggregator",
        category=NodeCategory.PROCESSING,
        binary="msight_launch_image_to_video_aggregator",
        invocation="supervisor",
        arg_map={
            "buffer_size": "--buffer-size",
            "overlap_size": "--overlap-size",
            "fps": "--fps",
        },
        default_name="video_aggregator",
    ),
    "video_local_dumper": NodeSpec(
        node_type="video_local_dumper",
        category=NodeCategory.SINK,
        binary="msight_launch_video_local_dumper",
        invocation="supervisor",
        required_config=("save_dir",),
        arg_map={"save_dir": "--save-dir"},
        default_name="local_dumper",
    ),
    "aws_video_pusher": NodeSpec(
        node_type="aws_video_pusher",
        category=NodeCategory.SINK,
        binary="msight_launch_aws_video_pusher",
        invocation="supervisor",
        required_config=("bucket_name",),
        arg_map={"bucket_name": "--bucket-name", "prefix": "--prefix"},
        default_name="s3_pusher",
    ),

    "aggregator": NodeSpec(
        node_type="aggregator", category=NodeCategory.PROCESSING,
        binary="msight_launch_aggregator", invocation="supervisor",
        required_config=("buffer_size", "overlap_size"),
        arg_map={"buffer_size": "--buffer-size", "overlap_size": "--overlap-size"},
    ),
    "aws_sequence_pusher": NodeSpec(
        node_type="aws_sequence_pusher", category=NodeCategory.SINK,
        binary="msight_launch_aws_sequence_pusher", invocation="supervisor",
        required_config=("bucket_name",),
        arg_map={"bucket_name": "--bucket-name", "prefix": "--prefix", "aws_region": "--aws-region"},
        positional_flags=("use_dualstack_endpoint",),
    ),
    "buffering_sort": NodeSpec(
        node_type="buffering_sort", category=NodeCategory.PROCESSING,
        binary="msight_launch_buffering_sort", invocation="supervisor",
        arg_map={"max_buffer_size": "--max-buffer-size"},
    ),
    "bytes_viewer": NodeSpec(
        node_type="bytes_viewer", category=NodeCategory.SINK,
        binary="msight_launch_bytes_viewer", invocation="supervisor",
        arg_map={"filter_sensor_name": "--filter-sensor-name"},
    ),
    "detection_results_viewer": NodeSpec(
        node_type="detection_results_viewer", category=NodeCategory.SINK,
        binary="msight_launch_detection_results_viewer", invocation="supervisor",
    ),
    "http_sink": NodeSpec(
        node_type="http_sink", category=NodeCategory.SINK,
        binary="msight_launch_http", invocation="supervisor",
        required_config=("url",),
        arg_map={
            "url": "--url", "partition_key_mode": "--partition-key-mode",
            "wait": "--wait", "shards": "--shards",
        },
    ),
    "ifm": NodeSpec(
        node_type="ifm", category=NodeCategory.SINK,
        binary="msight_launch_ifm", invocation="supervisor",
        required_config=("header_file", "rsu_addr", "rsu_port"),
        arg_map={"header_file": "--header-file", "rsu_addr": "--rsu-addr", "rsu_port": "--rsu-port"},
        positional_flags=("ipv6",),
    ),
    "image_viewer": NodeSpec(
        node_type="image_viewer", category=NodeCategory.SINK,
        binary="msight_launch_image_viewer", invocation="supervisor",
        arg_map={"filter_sensor_name": "--filter-sensor-name"},
    ),
    "local_image": NodeSpec(
        node_type="local_image", category=NodeCategory.SOURCE,
        binary="msight_launch_local_image", invocation="supervisor",
        required_config=("image_path",),
        arg_map={"image_path": "--image-path", "fps": "--fps"},
    ),
    "pointcloud_local_dumper": NodeSpec(
        node_type="pointcloud_local_dumper", category=NodeCategory.SINK,
        binary="msight_launch_pointcloud_local_dumper", invocation="supervisor",
        required_config=("output_folder_path",),
        arg_map={"output_folder_path": "--output-folder-path"},
        positional_flags=("use_creation_timestamp",),
    ),
    "pointcloud_viewer": NodeSpec(
        node_type="pointcloud_viewer", category=NodeCategory.SINK,
        binary="msight_launch_pointcloud_viewer", invocation="supervisor",
        arg_map={
            "filter_sensor_name": "--filter-sensor-name",
            "voxel_downsample": "--voxel-downsample", "color_mode": "--color-mode",
        },
    ),
    "sdsm_encoder": NodeSpec(
        node_type="sdsm_encoder", category=NodeCategory.PROCESSING,
        binary="msight_launch_sdsm_encoder", invocation="supervisor",
        required_config=("map_center", "source_id"),
        arg_map={
            "map_center": "--map-center", "source_id": "--source-id",
            "max_obj_list_length": "--max-obj-list-length",
        },
    ),
    "udp_server": NodeSpec(
        node_type="udp_server", category=NodeCategory.SOURCE,
        binary="msight_launch_udp_server", invocation="supervisor",
        required_config=("port",),
        arg_map={"host": "--host", "port": "--port"},
        positional_flags=("ipv6",),
    ),
    "velodyne_lidar": NodeSpec(
        node_type="velodyne_lidar", category=NodeCategory.SOURCE,
        binary="msight_launch_velodyne_lidar", invocation="supervisor",
        arg_map={
            "host": "--host", "port": "--port", "telemetry_port": "--telemetry-port",
            "model_id": "--model-id",
        },
        positional_flags=("ipv6",),
    ),
    "websocket_client": NodeSpec(
        node_type="websocket_client", category=NodeCategory.SOURCE,
        binary="msight_launch_websocket_client", invocation="supervisor",
        required_config=("server_url",),
        arg_map={"server_url": "--server-url"},
    ),
    "custom_fuser": NodeSpec(
        node_type="custom_fuser", category=NodeCategory.PROCESSING,
        binary="msight_launch_custom_fuser", invocation="supervisor",
        # FuserNode asserts sensor_name, so require it here for a clear error.
        required_config=("fusion_config", "sensor_name"),
        arg_map={"fusion_config": "--fusion-config", "wait": "--wait"},
    ),
    "finite_difference_state_estimator": NodeSpec(
        node_type="finite_difference_state_estimator", category=NodeCategory.PROCESSING,
        binary="msight_launch_finite_difference_state_estimator", invocation="supervisor",
        required_config=("estimator_configs",),
        arg_map={"estimator_configs": "--estimator-configs", "wait": "--wait"},
    ),
    "sort_tracker": NodeSpec(
        node_type="sort_tracker", category=NodeCategory.PROCESSING,
        binary="msight_launch_sort_tracker", invocation="supervisor",
        required_config=("tracking_configs",),
        arg_map={"tracking_configs": "--tracking-configs", "wait": "--wait"},
    ),
    "yolo_onestage_detection": NodeSpec(
        node_type="yolo_onestage_detection", category=NodeCategory.PROCESSING,
        binary="msight_launch_yolo_onestage_detection", invocation="supervisor",
        required_config=("det_configs",),
        arg_map={"det_configs": "--det-configs", "wait": "--wait"},
        # No needs_gpu: runs in the local venv and falls back to CPU on its own.
    ),
    "2d_viewer": NodeSpec(
        node_type="2d_viewer", category=NodeCategory.SINK,
        binary="msight_launch_2d_viewer", invocation="supervisor",
    ),
    "road_user_list_viewer": NodeSpec(
        node_type="road_user_list_viewer", category=NodeCategory.SINK,
        binary="msight_launch_road_user_list_viewer", invocation="supervisor",
        required_config=("basemap",),
        arg_map={"basemap": "--basemap"},
        positional_flags=("show_trajectory", "show_heading"),
    ),
}


def resolve_cmd(spec: NodeSpec, name: str, config: dict) -> list[str]:
    """Build a node's argv; binary path resolution is left to MSightControlPlane."""
    missing = [k for k in spec.required_config if k not in config]
    if missing:
        raise ValueError(
            f"node_type '{spec.node_type}' is missing required config: {missing}"
        )

    args = [spec.binary, "--name", name]

    if spec.category in (NodeCategory.SOURCE, NodeCategory.PROCESSING):
        if "publish_topic" not in config:
            raise ValueError(f"node_type '{spec.node_type}' requires 'publish_topic'")
        args += ["--publish-topic", config["publish_topic"]]
    if spec.category in (NodeCategory.PROCESSING, NodeCategory.SINK):
        if "subscribe_topic" not in config:
            raise ValueError(f"node_type '{spec.node_type}' requires 'subscribe_topic'")
        args += ["--subscribe-topic", config["subscribe_topic"]]
    if spec.category == NodeCategory.SOURCE:
        if "sensor_name" not in config:
            raise ValueError(f"node_type '{spec.node_type}' requires 'sensor_name'")
        args += ["--sensor-name", config["sensor_name"]]
    elif spec.category == NodeCategory.PROCESSING and "sensor_name" in config:
        args += ["--sensor-name", config["sensor_name"]]

    for key, flag in spec.arg_map.items():
        if key in config and config[key] not in (None, ""):
            args += [flag, str(config[key])]
    for key in spec.positional_flags:
        if config.get(key):
            args.append(f"--{key.replace('_', '-')}")

    return args
