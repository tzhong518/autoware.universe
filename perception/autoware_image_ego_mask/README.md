# autoware_image_ego_mask

Static 2D polygon masking node for camera images (intended for masking ego-vehicle regions such as hood/mirrors in side cameras).

## Topics

Topics depend on `input_image_base_topic` / `output_image_base_topic` (see below). Defaults are private `~/input/image` and `~/output/image`. With `image_transport`, the actual graph topics add a transport suffix (for example `.../compressed` when `input_transport` is `compressed`).

## Parameters

- `enabled` (bool, default: `true`)
- `input_transport` (string, default: `raw`): `image_transport` hint, e.g. `raw` or `compressed`.
- `output_transport` (string, default: `raw`)
- `input_image_base_topic` (string, default: empty): **image_transport base topic** without a `/compressed` suffix. If empty, the node uses `~/input/image`. Set this to an absolute topic (e.g. `/sensing/camera/camera1/image_raw`) so wiring does not rely on remaps (remaps often do not apply to `image_transport` plugin subscriptions).
- `output_image_base_topic` (string, default: empty): Same for output; default `~/output/image` when empty.
- `fill_value_bgr` (double[3], default: `[0,0,0]`): fill color inside mask polygons (BGR, 0–255).
- `polygons_yaml_file` (string, default: empty): path to YAML describing polygons. If non-empty, this file is read first.
- `polygons_yaml` (string, default: empty): inline YAML body for polygons. Used when `polygons_yaml_file` is empty.

Polygon geometry is defined only via YAML (file or inline). Example file: `config/example_polygons.yaml`.

### YAML schema

Root mapping:

```yaml
polygons:
  # Simple form: each entry is [x0, y0, x1, y1, ...] with at least 3 vertices (6 numbers).
  - [0.88, 0.0, 1.0, 0.0, 1.0, 1.0, 0.88, 1.0]
  # Or extended form with per-polygon flags:
  - points: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    normalized: false

# Optional; same length as polygons when present. Applies to simple-form entries.
polygons_normalized:
  - true
  - false
```

If `polygons_normalized` is omitted, coordinates are in pixels unless a polygon uses the map form with `normalized: true`.

### StreamPETR (camera9 / camera10)

Perception passes camera9 and camera10 into object detection as `image_raw7` and `image_raw8`. To insert ego masking for StreamPETR, enable it from Autoware launch:

```bash
ros2 launch autoware_launch autoware.main.launch.xml ... \
  use_streampetr_camera_ego_mask:=true \
  ego_mask_camera9_yaml:=/path/to/camera9_polygons.yaml \
  ego_mask_camera10_yaml:=/path/to/camera10_polygons.yaml \
  ego_mask_input_transport:=raw
```

Use `ego_mask_input_transport:=compressed` if those cameras publish `CompressedImage` only.

When enabled, two `ego_mask_node` instances subscribe to `/sensing/camera/camera9|camera10/image_raw` (plus transport) and publish `.../image_raw_ego_masked`; StreamPETR is wired to the masked topics automatically.

## Launch example (standalone node)

```bash
ros2 launch autoware_image_ego_mask ego_mask.launch.xml \
  input/image:=/sensing/camera/camera6/image_raw \
  output/image:=/sensing/camera/camera6/image_raw_masked \
  polygons_yaml_file:=/path/to/my_polygons.yaml
```

### Compressed input (e.g. rosbag)

Use the **base** topic `.../image_raw` plus `input_transport:=compressed`. Prefer parameters, not remaps:

```bash
ros2 run autoware_image_ego_mask ego_mask_node --ros-args \
  -p input_transport:=compressed \
  -p input_image_base_topic:=/sensing/camera/camera1/image_raw \
  -p output_image_base_topic:=/debug/camera1/masked \
  -p polygons_yaml_file:=/path/to/polygons.yaml
```

With rosbag and `/clock`, also set `use_sim_time:=true` on the node and play the bag with `--clock` if needed.

To override with inline YAML from the shell, pass an empty file argument and set `polygons_yaml` via a params file or another launch mechanism (multiline YAML in `ros2 param set` is awkward; prefer a file for production).
