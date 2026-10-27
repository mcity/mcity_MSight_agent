"""Generic add/remove node tools for building custom topologies.

Thin wrappers over MSightControlPlane; not consent-gated since single-node
changes are cheap and reversible.
"""
import difflib
from typing import Optional

from msight_agent.shared.mcp import error_json, mcp, ok_json
from msight_agent.msight.runtime.node_catalog import (
    NODE_CATALOG, missing_config, unknown_config, valid_config_keys, with_defaults,
)
from msight_agent.msight.runtime.control_plane import get_control_plane
from msight_agent.msight.runtime.utils import next_free_port, port_in_use, topic_list
from msight_agent.shared.host_utils import resolve_host
from msight_agent.msight.runtime.paths import get_msight_path
from msight_agent.msight.tools.pipeline import describe_by_role, node_role

_NODE_TYPE_DESCRIPTIONS: dict[str, str] = {
    "video_source_rtsp": "Publishes frames from a live RTSP stream (or a local video file path passed as rtsp_url).",
    "video_source_mp4_folder": "Publishes frames sequentially from a folder of .mp4 files.",
    "rfdetr_detector": "Runs RF-DETR object detection on a frame topic, publishes detections. Uses the GPU; on a GPU-less host it falls back to a much slower CPU-only image.",
    "detection_viewer": "Web viewer (configurable port) that renders a detection topic as annotated video in-browser.",
    "frame_annotator": "Draws detection boxes/labels onto frames, republishes as an annotated frame topic.",
    "video_aggregator": "Buffers an annotated frame topic into short video clips.",
    "video_local_dumper": "Writes a video topic's clips to local disk.",
    "aws_video_pusher": "Pushes a video topic's clips to an S3 bucket.",
    "aggregator": "Generic frame-buffering aggregator (distinct from video_aggregator's image-to-video specialization).",
    "aws_sequence_pusher": "Pushes a raw data sequence topic to an S3 bucket.",
    "http_sink": "POSTs a topic's messages to an external HTTP endpoint.",
    "ifm": "Pushes data to a roadside unit (RSU) over IFM. Needs a real, reachable RSU device.",
    "image_viewer": "Renders an image topic in a GUI window (cv2.imshow) -- needs a real display/X11, not headless-safe.",
    "local_image": "Replays a single local image file as a source topic, at a configurable fps -- simple test/dummy source.",
    "pointcloud_local_dumper": "Writes a point-cloud topic's data to local disk.",
    "pointcloud_viewer": "Renders a point-cloud topic in a 3D GUI window (open3d) -- needs a real display, not headless-safe.",
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
            "defaults": spec.default_config,
            # Wire a node only to a topic whose data kind it consumes; add_msight_node enforces it.
            **({"consumes": list(spec.consumes)} if spec.consumes else {}),
            **({"produces": spec.produces} if spec.produces else {}),
            "needs_gpu": spec.needs_gpu,
            "default_name": spec.default_name,
        }
        for node_type, spec in sorted(NODE_CATALOG.items())
    ]
    return ok_json(node_types=catalog)


def _input_topic_problem(config: dict, published: set[str], described: Optional[str] = None) -> Optional[str]:
    """A subscribe_topic nothing publishes, while other topics are flowing, is almost
    always a guessed or mistyped name. (An empty pipeline is allowed: nodes may be
    added in any order.)"""
    if not published:
        return None
    for topic in topic_list(config.get("subscribe_topic")):
        if topic not in published:
            close = difflib.get_close_matches(topic, sorted(published), n=1, cutoff=0.5)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            return (f"No running node publishes '{topic}'.{hint} Topics it can subscribe to right now: "
                    f"{described or sorted(published)}. Use one of these as subscribe_topic, or add the "
                    "node that publishes it first.")
    return None


def _output_topic_problem(config: dict, published: set[str], name: str) -> Optional[str]:
    """A second publisher on a live topic interleaves two nodes' output on it (e.g. two
    detectors' boxes flickering in one viewer), and no health check can tell."""
    taken = [t for t in topic_list(config.get("publish_topic")) if t in published]
    if not taken:
        return None
    return (f"'{taken[0]}' already has a running publisher -- a second one would mix both nodes' output "
            f"on that topic. Give this node its own publish_topic (e.g. '{taken[0]}_{name}'), and point "
            f"any viewer for it at that topic. If the user asked for '{taken[0]}' themselves, tell them "
            "it's taken and which topic you used instead.")


def _resolve_node_names(config: dict, node_topics: dict[str, list[str]]) -> tuple[dict, list[str]]:
    """A subscribe_topic that is really a running node's name ("video_source") means the one
    topic that node publishes. Returns (config, notes saying what was substituted)."""
    topics = topic_list(config.get("subscribe_topic"))
    resolved, notes = [], []
    for t in topics:
        published = node_topics.get(t, [])
        if len(published) == 1 and t not in {x for ts in node_topics.values() for x in ts}:
            resolved.append(published[0])
            notes.append(f"'{t}' is a node, not a topic -- subscribed to '{published[0]}', the topic it publishes.")
        else:
            resolved.append(t)
    if not notes:
        return config, []
    value = resolved[0] if isinstance(config.get("subscribe_topic"), str) else resolved
    return {**config, "subscribe_topic": value}, notes


# Topic prefix for a default output topic, by the kind of data a node produces.
_OUTPUT_PREFIX = {"detections": "detection", "video": "video", "sequence": "aggregated",
                  "image": "annotated", "bytes": "bytes", "pointcloud": "pointcloud"}


def _fill_wiring_defaults(spec, config: dict, name: str,
                          publisher_kinds: dict[str, tuple[str, str, str]]) -> tuple[dict, list[str]]:
    """Fill topics only where there's no real choice: an input when exactly one running topic
    fits, an output named after the node. Returns (config, notes saying what was chosen)."""
    config, notes = dict(config), []
    takes_input = spec.category.value in ("processing", "sink")
    if takes_input and "subscribe_topic" not in config and spec.consumes:
        fitting = [t for t, (kind, _, _) in publisher_kinds.items() if kind in spec.consumes]
        if len(fitting) == 1:
            config["subscribe_topic"] = fitting[0]
            notes.append(f"Subscribed to '{fitting[0]}', the only {'/'.join(spec.consumes)} topic running.")
    if spec.category.value == "processing" and "publish_topic" not in config and spec.produces:
        config["publish_topic"] = f"{_OUTPUT_PREFIX.get(spec.produces, spec.produces)}/{name}"
        notes.append(f"Publishing to '{config['publish_topic']}'.")
    return config, notes


def _describe_topics(publisher_kinds: dict[str, tuple[str, str, str]], spec=None) -> str:
    """The topics a node could subscribe to, with the data each carries. For a node that
    reads a known kind, only the topics of that kind -- so a wrong one can't be offered."""
    topics = sorted(publisher_kinds.items())
    if spec is not None and spec.consumes:
        fitting = [t for t, (kind, _, _) in topics if kind in spec.consumes]
        reads = "/".join(spec.consumes)
        if not fitting:
            return f"none with the {reads} data '{spec.node_type}' reads -- add a node that publishes it first"
        return f"{', '.join(fitting)} (the only {reads} topics, which is what '{spec.node_type}' reads)"
    return ", ".join(f"{t} ({kind or 'unknown data'})" for t, (kind, _, _) in topics)


def _input_kind_problem(spec, config: dict, publisher_kinds: dict[str, tuple[str, str, str]]) -> Optional[str]:
    """A node fed a data kind it can't read crashes on its first message (e.g. a detection
    viewer on a raw camera topic). `publisher_kinds`: topic -> (kind, publisher, its type)."""
    if not spec.consumes:
        return None
    for topic in topic_list(config.get("subscribe_topic")):
        kind, publisher, publisher_type = publisher_kinds.get(topic, (None, "", ""))
        if kind and kind not in spec.consumes:
            fitting = sorted(t for t, (k, _, _) in publisher_kinds.items() if k in spec.consumes)
            return (f"'{spec.node_type}' reads {'/'.join(spec.consumes)} data, but '{topic}' carries {kind} "
                    f"(published by '{publisher}', a {publisher_type}); it would crash on the first message. "
                    "If the user chose that topic, tell them why it won't work before using another. "
                    + (f"Topics with {'/'.join(spec.consumes)} data right now: {fitting}." if fitting
                       else f"Nothing publishes {'/'.join(spec.consumes)} data right now -- add a node that does first."))
    return None


def _resolve_web_port(spec, config: dict) -> tuple[dict, Optional[str]]:
    """Fill in a free port for web nodes that weren't given one; refuse a port that's taken
    (the web server would fail to bind while the node keeps running, looking healthy)."""
    if not spec.web_port:
        return config, None
    if "port" not in config:
        return {**config, "port": next_free_port(spec.web_port)}, None
    if port_in_use(int(config["port"])):
        return config, (f"Port {config['port']} is already in use on this host. Pick another port "
                        f"(e.g. {next_free_port(int(config['port']) + 1)}) or leave port out to get a free one.")
    return config, None


@mcp.tool()
async def add_msight_node(node_type: str, name: Optional[str] = None, config: Optional[dict] = None) -> str:
    if err := get_msight_path()[1]:
        return error_json(err)
    if node_type not in NODE_CATALOG:
        return error_json(f"Unknown node_type '{node_type}'. Call list_msight_node_types() first.")

    spec = NODE_CATALOG[node_type]
    resolved_name = name or spec.default_name or node_type
    cp = get_control_plane()

    # A live node with this name is reused untouched. Don't go through add_node: a node
    # this process doesn't track (e.g. from before a restart) would be recreated from
    # scratch, losing its config -- including its web port.
    live = (await cp.snapshot([resolved_name]))[resolved_name]
    if live["alive"]:
        port = live.get("web_port")
        url = f"http://{resolve_host()}:{port}" if port else None
        return ok_json(
            f"'{resolved_name}' was already running -- reused as-is, no new node was "
            f"started (add_msight_node is idempotent by name)."
            + (f" It serves {url}." if url else "")
            + " If you wanted a SEPARATE node, call again with a different `name`.",
            name=resolved_name, node_type=node_type, already_running=True, **({"url": url} if url else {}),
        )

    config = with_defaults(spec, dict(config or {}))
    if unknown := unknown_config(spec, config):
        return error_json(f"{unknown} {'is not a setting' if len(unknown) == 1 else 'are not settings'} of "
                          f"'{node_type}' and would be ignored. Valid settings: {sorted(valid_config_keys(spec))}.")
    registry = cp.get_status()
    started = cp.started_publishers()  # includes nodes still starting up (not registered yet)
    published = {t for n in registry for t in topic_list(n.get("publish_topic"))} | set(started)

    # Only live publishers count (registry entries can outlive a crashed node).
    snap = await cp.snapshot(n["name"] for n in registry)
    publisher_kinds = {}
    for n in registry:
        if not snap[n["name"]]["alive"]:
            continue
        publisher_type = await cp.node_type_of(n["name"])
        kind = NODE_CATALOG[publisher_type].produces if publisher_type in NODE_CATALOG else None
        for t in topic_list(n.get("publish_topic")):
            publisher_kinds[t] = (kind, n["name"], publisher_type)
    for t, (publisher, publisher_type) in started.items():
        if t not in publisher_kinds and await cp.is_alive(publisher):
            publisher_kinds[t] = (NODE_CATALOG[publisher_type].produces, publisher, publisher_type)

    node_topics = {n["name"]: topic_list(n.get("publish_topic")) for n in registry if snap[n["name"]]["alive"]}
    config, resolved_notes = _resolve_node_names(config, node_topics)
    config, default_notes = _fill_wiring_defaults(spec, config, resolved_name, publisher_kinds)
    resolved_notes += default_notes

    # Wiring problems first: they're about choices already made (a wrong topic is worth
    # reporting before asking for the settings still missing).
    if problem := (_input_topic_problem(config, published, _describe_topics(publisher_kinds, spec) or None)
                   or _input_kind_problem(spec, config, publisher_kinds)
                   or _output_topic_problem(config, set(publisher_kinds), resolved_name)):
        return error_json(problem)
    if missing := missing_config(spec, config):
        hint = (f" Topics it can subscribe to right now: {_describe_topics(publisher_kinds, spec)}."
                if publisher_kinds and "subscribe_topic" in missing else "")
        return error_json(f"'{node_type}' needs {missing} in config (all at once, in one call).{hint}")
    config, problem = _resolve_web_port(spec, config)
    if problem:
        return error_json(problem)

    try:
        await cp.add_node(node_type, name=name, config=config)
    except Exception as e:
        return error_json(f"Could not start '{resolved_name}' ({node_type}): {e}")

    extra = {"url": f"http://{resolve_host()}:{config['port']}"} if spec.web_port else {}
    return ok_json(
        f"'{resolved_name}' ({node_type}) started." + (f" Open it at {extra['url']}." if extra else "")
        + "".join(f" {note}" for note in resolved_notes),
        name=resolved_name, node_type=node_type, already_running=False, **extra,
    )


@mcp.tool()
async def remove_msight_node(name: str) -> str:
    if err := get_msight_path()[1]:
        return error_json(err)
    cp = get_control_plane()
    existed = cp.is_tracked(name) or await cp.is_alive(name)
    try:
        await cp.delete_node(name)
        still_running = await cp.live_node_names()
    except Exception as e:
        return error_json(f"Could not stop '{name}': {e}")
    # Listed so a "remove everything" request can't silently skip one.
    return ok_json(
        (f"'{name}' stopped." if existed else f"'{name}' wasn't running; nothing to stop.")
        + (f" Still running -- {describe_by_role(still_running)}." if still_running
           else " No MSight nodes are running now.")
        + (" That was a standard pipeline node, so the pipeline is now incomplete." if node_role(name) == "pipeline"
           else ""),
        name=name, still_running=still_running,
    )
