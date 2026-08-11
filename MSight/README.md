# MSight — Localization Module

Offline localization pipeline for camera-based object detection stored in [FiftyOne](https://docs.voxel51.com/) datasets.
Ported from `mcity_data_engine_msight` (`aws-agentic-dataengine` branch) and integrated into the Auto Labeling workflow.

Given a FiftyOne dataset with 2-D bounding-box detections and an NPZ camera-calibration file, the pipeline:
1. Converts each detection to an MSight `DetectedObject2D`.
2. Estimates the fisheye ground-contact pixel using camera intrinsics.
3. Maps pixel coordinates to lat/lon via interpolation over calibration control points.
4. Writes localized detections and keypoints back to the FiftyOne dataset.

---

## Directory structure

```
MSight/
├── data/
│   ├── ashley_huron_intrinsic.json          # Hardcoded camera intrinsics (see Configuration below)
│   └── calibration_results_ashley_huron.npz # Hardcoded lat/lon calibration grid
├── utils/
│   ├── __init__.py
│   ├── fiftyone_to_msight_det.py            # FiftyOne → MSight detection conversion
│   └── load_locamaps.py                     # NPZ loader, intrinsics loader, pixel localizer
├── localize_dataset.py                      # Core localization logic + FiftyOne write-back
├── install.sh                               # Install script for MSight dependencies
├── requirements.txt                         # Python package requirements
└── README.md
```

---

## Installation

Install MSight dependencies into the **active** Python environment before enabling localization:

```bash
# Activate your project virtual environment first (if applicable)
source venv/bin/activate

# Then run the install script
bash MSight/install.sh
```

Alternatively install directly with pip:

```bash
pip install -r MSight/requirements.txt
```

### Key dependencies

| Package | Purpose |
|---------|---------|
| `msight_base` | `DetectedObjectBase`, `DetectionResultBase` |
| `msight_core` | MSight messaging infrastructure |
| `scipy` | `LinearNDInterpolator` / `NearestNDInterpolator` for sparse-map localization |
| `numpy` | NPZ map loading and array operations |

---

## Configuration

Like the source repo's static `MSIGHT_CONFIG` dict in `config/config.py`, calibration here
is **hardcoded to one camera** (the Ashley/Huron intersection) — this is a known limitation,
not a design goal; see `msight-customizable-pipeline.md` for the planned per-dataset fix.
Only `detection_field` and `enabled` are runtime-configurable, via the chat agent's Auto
Labeling workflow's `set_msight_localization_config` MCP tool, which writes into
`config.WORKFLOWS["auto_labeling"]` (the same dict `main.py` already reads for
hyperparameters):

| Key | Description |
|-----|-------------|
| `localization_enabled` | `True` to run localization as part of the same `main.py` run that produces predictions — runtime-configurable |
| `localization_detection_field` | FiftyOne field holding `fo.Detections` to localize — runtime-configurable |
| `localization_intrinsics_path` | Hardcoded to `MSight/data/ashley_huron_intrinsic.json` — not settable via the tool |
| `localization_locmap_path` | Hardcoded to `MSight/data/calibration_results_ashley_huron.npz` — not settable via the tool |

Running localization against a dataset filmed by a different camera will silently apply the
Ashley/Huron calibration to it, producing incorrect lat/lon — there is no mechanism (here or
in the source repo) that picks calibration files based on the dataset.

`main.py`'s `_run_msight_localization()` reads these at the same point in execution the
source repo's version checked `MSIGHT_CONFIG['run_localization']` — right after the
dataset is saved, before the Voxel51 session launches.

---

## Camera intrinsics format

```json
{ "f": 320, "x0": 645, "y0": 473 }
```

| Key | Description |
|-----|-------------|
| `f`  | Focal length in pixels |
| `x0` | Pixel column of the fisheye circle centre |
| `y0` | Pixel row of the fisheye circle centre |

---

## Calibration file format

Each `.npz` must contain two arrays of shape `(H, W)` matching the camera resolution:

| Array | dtype | Description |
|-------|-------|-------------|
| `lat_map` | float64 | Latitude at each pixel (non-calibrated pixels are `-inf`) |
| `lon_map` | float64 | Longitude at each pixel (non-calibrated pixels are `-inf`) |

**This is a different key convention than the live MSight_Vision pipeline's calibration
files**, which use `x_map`/`y_map` (see `mcp_layer/mcptools/msight_docker.py`'s
`check_msight_calibration_status`). The two are not interchangeable — `set_msight_localization_config`
validates the `.npz` contains `lat_map`/`lon_map` before writing anything, and rejects
with a message naming the mismatch if it looks like the other format instead.

The pipeline automatically handles sparse calibration maps via `LinearNDInterpolator`
with a `NearestNDInterpolator` fallback.

---

## Output fields

The pipeline writes **two** new fields per sample to the FiftyOne dataset:

| Field | Type | Description |
|-------|------|-------------|
| `msight_<detection_field>` | `fo.Detections` | Localized detections with `lat`/`lon` attributes |
| `msight_<detection_field>_keypoints` | `fo.Keypoints` | One keypoint per detection at the fisheye ground-contact pixel |

---

## Fisheye ground-contact logic

For a fisheye camera mounted overhead, "down" in the image is radially inward toward the
optical centre `(x0, y0)`. The ground-contact pixel is the bounding-box boundary point along
the outward radial ray from `(x0, y0)` through the box centre — implemented in
`localize_dataset.fisheye_ground_contact(bbox_xyxy, x0, y0)`.

---

## Module reference

### `utils/load_locamaps.py`

| Function | Description |
|----------|-------------|
| `load_intrinsics(path)` | Loads camera intrinsics JSON; returns `{'f', 'x0', 'y0'}`. |
| `load_locmaps(path)` | Loads NPZ calibration file; returns `(lat_map, lon_map)`. |
| `build_pixel_localizer(lat_map, lon_map)` | Returns a callable `localize(cx, cy) -> (lat, lon)`. |

### `utils/fiftyone_to_msight_det.py`

Converts `fo.Detections` to `DetectionResultBase` containing `DetectedObject2D` objects.

### `localize_dataset.py`

| Function | Description |
|----------|-------------|
| `fisheye_ground_contact(bbox_xyxy, x0, y0)` | Returns fisheye-corrected ground-contact pixel. |
| `localize_detection_result(...)` | Fills lat/lon on each detection via the localizer. |
| `build_fo_detections_from_msight(...)` | Converts localized `DetectionResultBase` to `fo.Detections`. |
| `build_fo_keypoints_from_msight(...)` | Builds `fo.Keypoints` from contact pixels. |
| `run_localization(...)` | Iterates every sample; writes detections and keypoints fields. |
