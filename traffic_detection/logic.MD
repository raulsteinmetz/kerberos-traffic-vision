## Subscribed Topics

| Topic                                | Message Type                 | Purpose                                                       |
|--------------------------------------|-------------------------------|---------------------------------------------------------------|
| `/flag/detection_switch`             | `std_msgs/String`            | `"Start"` to activate detection, `"Stop"` to pause it         |
| `/flag/load_yolo`                    | `std_msgs/Bool`              | `True` loads YOLO models; `False` unloads them from memory    |
| `/usb_cam1/image_raw/compressed`     | `sensor_msgs/CompressedImage`| Receives compressed image stream for detection (via YOLO)     |


## Published Topics

| Topic                                | Message Type                    | Purpose                                                                 |
|--------------------------------------|----------------------------------|-------------------------------------------------------------------------|
| `car_on_crosswalk`                   | `std_msgs/Bool`                 | `True` = safe to proceed (no car on zebra); `False` = car detected     |
| `traffic_light_state`                | `std_msgs/String`               | `"start"` if green seen ≥ 20 times; otherwise `"stop"` (red/unknown)   |
| `detection_image`                    | `sensor_msgs/Image`             | Publishes an annotated debug image with bounding boxes                 |
| `marker_array`                       | `visualization_msgs/MarkerArray`| Publishes RViz markers for all detected bounding boxes (cars/lights)   |
