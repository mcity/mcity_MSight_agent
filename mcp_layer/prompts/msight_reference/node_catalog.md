# MSight node catalog reference

Every entry below is a `node_type` `add_msight_node` accepts today (confirmed
against `mcp_layer/mcptools/msight_node_catalog.py` — 29 node types
cataloged). 3 real msight_core entry points are deliberately excluded, not
just unlisted: `downloaded_image_player_fps` and `road_user_list_aggregator`
(their source files don't exist in the installed package — genuinely broken
today) and `downloaded_image_player` (exists, but references an argument it
never defines — guaranteed to crash). If a node type comes up that isn't
here and isn't one of these three, say so plainly rather than guessing a
name; it needs a new `NodeSpec` entry (a code change), not a config you're
missing.

Every node needs `--name` (the `name` argument to `add_msight_node`, defaults
to the catalog's `default_name` if omitted). Beyond that, config keys depend
on category:
- **source** nodes need `publish_topic` and `sensor_name`.
- **processing** nodes need `publish_topic` and `subscribe_topic` (`sensor_name` optional).
- **sink** nodes need `subscribe_topic`.

Calibration is handled separately from node config, not through this
catalog — `check_msight_calibration_status` reports it, and the upload flow
described in the msight_pipeline workflow prompt sets it. It matters for
this catalog because `rfdetr_detector`'s output (and anything downstream of
it, like a fuser) is only meaningful in world terms once calibration is in
place — a camera source producing detections with default/missing
calibration is a config state worth checking, not a node problem.

## video_source_rtsp (source)
Publishes frames from a live RTSP stream — or a local video **file** path
passed as `rtsp_url` (the same flag doubles as a plain file path when it
isn't a URL). For a **folder** of files, use `video_source_mp4_folder`
instead.

Required config: `rtsp_url`. Optional: `rtsp_transport`.
```
add_msight_node(node_type="video_source_rtsp", name="video_source_2", config={
  "publish_topic": "camera/gs_mcity_2", "sensor_name": "gs_mcity_2",
  "rtsp_url": "rtsp://10.0.1.70:9001/1"
})
```

## video_source_mp4_folder (source)
Publishes frames sequentially from a folder of `.mp4` files. Required
config: `folder` (host path, gets bind-mounted automatically). Optional:
`resize_ratio`.

## rfdetr_detector (processing, needs GPU)
Runs RF-DETR object detection on a frame topic, publishes detections.
Required config: `det_configs` (a path *inside the container* — normally
`/configs/rfdetr_config.yaml`, already bind-mounted; don't pass a host
path here). Gotcha: this node type will fail to start on a host with no
GPU (`nvidia-smi` unavailable) — see common_failure_modes.md.

## detection_viewer (sink)
Web viewer that renders a detection topic as annotated video in-browser.
No required config beyond `subscribe_topic`; optional `port` (defaults to
9010). **Always pass an explicit, distinct `name` when adding a second one**
— omitting `name` defaults to `"detection_viewer"`, the same name the main
pipeline's own viewer already uses, and `add_msight_node` is idempotent by
name: it will silently reuse the existing node (ignoring your `port` config
entirely) instead of starting a new one. Confirmed happening live — the
symptom is the response still says success, but nothing new is actually
listening on the port you asked for. Correct form:
`add_msight_node(node_type="detection_viewer", name="viewer_2", config={"subscribe_topic": "detection/gs_mcity_2", "port": 9011})`.

## frame_annotator (processing)
Draws detection boxes/labels onto frames, republishes as an annotated frame
topic. This is the agent's own script, not an installed msight_core binary
— runs via the Supervisor backend, never Docker. No `sensor_name` flag
exists for this one; don't pass it.

## video_aggregator (processing)
Buffers an annotated frame topic into short video clips (used by both
recording and archiving — see `start_msight_recording`/`start_msight_archiving`
for the normal path; add this directly only for a non-standard topology).
Optional config: `buffer_size`, `overlap_size`, `fps`.

## video_local_dumper (sink)
Writes a video topic's clips to local disk. Required config: `save_dir`.

## aws_video_pusher (sink)
Pushes a video topic's clips to an S3 bucket. Required config:
`bucket_name`. Optional: `prefix`. Needs `AWS_ACCESS_KEY_ID`/
`AWS_SECRET_ACCESS_KEY` set in `.env` — check first if unsure.

---

## The rest of the catalog

All of the following run via the Supervisor backend (MSight_Vision's local
venv directly, not Docker) — no volume-mount handling needed even for the
ones that take local file paths, since a bare process already sees the
whole host filesystem.

### aggregator (processing)
Generic frame-buffering aggregator — distinct from `video_aggregator`
(image-to-video specific). Required config: `buffer_size`, `overlap_size`.

### aws_sequence_pusher (sink)
Pushes a raw data-sequence topic to S3. Required: `bucket_name`. Optional:
`prefix`, `aws_region`, `use_dualstack_endpoint` (boolean flag).

### buffering_sort (processing)
Buffers and time-sorts an out-of-order topic. Optional: `max_buffer_size`.

### bytes_viewer (sink)
Logs raw bytes-topic messages as text — no GUI, safe for headless hosts.
Optional: `filter_sensor_name`.

### detection_results_viewer (sink)
Logs detection results as text — no GUI, safe for headless hosts. No extra
config beyond `subscribe_topic`. (Don't confuse with `2d_viewer` below,
which renders the same kind of data in an actual GUI window.)

### http_sink (sink)
POSTs a topic's messages to an external HTTP endpoint. Required: `url`.
Optional: `partition_key_mode` (`random` or `sensor_name`), `wait`,
`shards`. Needs a real, reachable endpoint.

### ifm (sink)
Pushes data to a roadside unit (RSU) over IFM. Required: `header_file`,
`rsu_addr`, `rsu_port`. Optional: `ipv6` (boolean flag). Needs a real,
reachable RSU device on the network — not something you can test without one.

### image_viewer (sink)
Renders an image topic via `cv2.imshow`. **Needs a real display/X11** —
will not work on a headless host. Optional: `filter_sensor_name`. If asked
for this on a headless deployment, say so rather than adding it and letting
it fail silently.

### local_image (source)
Replays a single local image file as a source topic at a fixed fps — a
simple test/dummy source, not a real camera feed. Required: `image_path`
(a real file on this host). Optional: `fps`.

### pointcloud_local_dumper (sink)
Writes a point-cloud topic's data to local disk. Required:
`output_folder_path`. Optional: `use_creation_timestamp` (boolean flag).

### pointcloud_viewer (sink)
Renders a point-cloud topic in a 3D GUI window via `open3d`. **Needs a
real display** — will not work on a headless host, same caveat as
`image_viewer`. Optional: `filter_sensor_name`, `voxel_downsample`,
`color_mode` (`ring` or `intensity`).

### sdsm_encoder (processing)
Encodes detections into SDSM (Sensor Data Sharing Message) format.
Required: `map_center` (a `"(lat, lon)"`-shaped string — it's parsed as a
Python literal, so it must look exactly like a tuple, not free text),
`source_id`. Optional: `max_obj_list_length`.

### udp_server (source)
Listens for incoming UDP packets, publishes them as a source topic.
Required: `port`. Optional: `host` (default `0.0.0.0`), `ipv6`. Needs an
external sender actually pushing packets to that port — adding this alone
produces a source that publishes nothing until one exists.

### velodyne_lidar (source)
Ingests a real Velodyne LiDAR unit's UDP stream. Optional: `host`, `port`,
`telemetry_port`, `model_id` (one of `HDL64E_S1/S2/S3`, `HDL32E`,
`VLP32A/B/C`, `VLP16`, `PuckLite`, `PuckHiRes`, `VLS128`, `AlphaPrime`).
**Needs real LiDAR hardware** (or a UDP packet replay of one) sending to
`host:port` — this will sit idle forever without one, not error out.

### websocket_client (source)
Connects to an external WebSocket server, publishes received messages as a
source topic. Required: `server_url`. Needs a real, reachable server.

### custom_fuser (processing)
Multi-sensor fusion, driven by a YAML fusion-config file. Required:
`fusion_config` (a real file on this host, referencing a fuser class and
sensor list), **and `sensor_name`** — unlike other processing nodes this
one hard-crashes without it, so it's enforced as required here too.

**Important limitation, not a config problem: there is no live localization
node.** MSight's fusion (`HungarianFuser`, what this node wraps) matches
objects *in world coordinates* — but no `msight_core` node exists anywhere
that converts a detection's pixel position into world lat/lon in the live
pub/sub pipeline (`HashLocalizer` only exists as a batch-script class, and
separately as an unrelated, offline, per-dataset step inside the Auto
Labeling workflow — neither applies here). If asked to set up live
multi-camera fusion, say plainly that the localization step it depends on
doesn't exist as a live node today — don't invent a `node_type` for it or
guess at a `custom_fuser` config that expects world coordinates it will
never receive.

### finite_difference_state_estimator (processing)
Estimates velocity/heading via finite differences, driven by a YAML
config file. Required: `estimator_configs` (a real file on this host).

### sort_tracker (processing)
SORT multi-object tracker, driven by a YAML config file (keys used:
`max_age`, `min_hits`, `iou_threshold`, `use_filtered_position`). Required:
`tracking_configs` (a real file on this host).

### yolo_onestage_detection (processing)
YOLO object detection, driven by a YAML config file naming the model.
Required: `det_configs` (a real file on this host). Uses GPU automatically
if present (checked live via the local venv, not a fixed Docker image) —
unlike `rfdetr_detector`, no CPU/GPU image swap is needed either way; it
just runs slower on CPU.

### 2d_viewer (sink)
Renders detection results via `cv2.imshow`. **Needs a real display/X11** —
same caveat as `image_viewer`/`pointcloud_viewer`. No extra config beyond
`subscribe_topic`.

### road_user_list_viewer (sink)
Renders a road-user list over a basemap image via `cv2.imshow`. **Needs a
real display/X11.** Required: `basemap` (a real image file on this host).
Optional: `show_trajectory`, `show_heading` (boolean flags).
