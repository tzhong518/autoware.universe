# autoware_image_ego_mask

Static 2D polygon masking node for camera images (intended for masking ego-vehicle regions such as hood/mirrors in side cameras).

## Topics

- **Input**: `~/input/image` (`sensor_msgs/msg/Image`)
- **Output**: `~/output/image` (`sensor_msgs/msg/Image`)

## Parameters

- `enabled` (bool, default: `true`)
- `fill_value_bgr` (double[3], default: `[0,0,0]`): fill color inside mask polygons.
- `polygons` (double[][]): list of polygons; each polygon is flattened `[x0,y0,x1,y1,...]`.
- `polygons_normalized` (bool[]): optional; if `true`, polygon points are interpreted as normalized ratios in \([0,1]\) and scaled by image width/height.

## Launch example

```bash
ros2 launch autoware_image_ego_mask ego_mask.launch.xml \
  input/image:=/sensing/camera/camera6/image_raw \
  output/image:=/sensing/camera/camera6/image_raw_masked \
  polygons:="[[0.88,0.0, 1.0,0.0, 1.0,1.0, 0.88,1.0]]" \
  polygons_normalized:="[true]"
```

