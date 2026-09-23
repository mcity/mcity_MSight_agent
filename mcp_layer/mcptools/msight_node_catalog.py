"""Data-driven catalog of msight_core node types this agent knows how to run.

Every msight_core node CLI shares a base argument parser
(msight_core.utils.get_default_arg_parser): --name is always required, and
--publish-topic / --subscribe-topic / --sensor-name are added per category
(source: publish-topic + required sensor-name; processing: publish-topic +
subscribe-topic + optional sensor-name; sink: subscribe-topic only) --
confirmed by reading MSight_Vision/venv/.../msight_core/utils.py directly, not
inferred. resolve_cmd() builds those automatically so NodeSpec only needs to
declare what's specific to that node type.

Only the 6 node types already wired today (as docker-compose.yml services or
as msight_record_archive.py's hardcoded subprocesses) are catalogued here.
Adding one of the ~25 other node types msight_core ships is meant to be a new
NodeSpec entry, not new code -- but verify its real CLI flags first by reading
MSight_Vision/cli/launch_*.py or the matching file under
MSight_Vision/venv/.../site-packages/cli/, the way the 6 below were verified.
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
    # If set (docker invocation only): when config[mount_config_key] is a host
    # path (not a URL), mount it read-only at mount_container_path inside the
    # container and rewrite the flag's value to that container path -- mirrors
    # docker-compose.yml's own `${VIDEO_INPUT:-/tmp}:/input:ro` bind mount.
    mount_config_key: str | None = None
    mount_container_path: str | None = None
    # supervisor-only: run `venv_python3 <script_path> <flags>` instead of a
    # venv binary directly -- for a node that's the agent's own script (e.g.
    # frame_annotator), not one of msight_core's installed console scripts.
    script_path: Path | None = None
    # Mounts that are always applied for this node type, independent of any
    # per-call config -- (path relative to MSIGHT_VISION_PATH, container path,
    # "ro"|"rw"). E.g. rfdetr_detector's config/calibration/locmap/model dirs,
    # which docker-compose.yml bind-mounts unconditionally.
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
        # msight_launch_rtsp also accepts a local file path as --url (a single
        # video file, as opposed to a folder) -- mount it if it's not a URL.
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
        # This agent's own script -- not one of msight_core's console scripts,
        # and its argparse has no --sensor-name flag, so no sensor_name is
        # ever passed in its config (resolve_cmd only adds that flag when the
        # config key is present).
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
        # sensor_name is optional for PROCESSING nodes per the base parser,
        # but FuserNode's own __init__ hard-asserts it's set -- required here
        # so a missing one is a clear error, not a container crash.
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
        # Not marked needs_gpu -- unlike rfdetr_detector this runs via the
        # local venv (not a fixed docker image), so it can use whatever GPU
        # is actually present via its own torch.cuda.is_available() check,
        # and degrades to CPU gracefully with no image swap needed either way.
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
    """Build the argv (binary name + flags) for one node, minus any path
    resolution -- MSightControlPlane decides whether `spec.binary` needs
    resolving to an absolute path (supervisor) or is already on PATH inside
    the image (docker) before handing this to an Executor."""
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
