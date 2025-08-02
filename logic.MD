## Subscribed Topics

| Topic                             | Message Type                   | Purpose                                                                |
|----------------------------------|--------------------------------|------------------------------------------------------------------------|
| `/flag/detection_switch`         | `std_msgs/String`              | `"Start"` to activate detection, `"Stop"` to pause it                  |
| `/flag/load_yolo`                | `std_msgs/Bool`                | `True` loads YOLO models, `False` unloads them from memory             |
| `/usb_cam1/image_raw/compressed` | `sensor_msgs/CompressedImage`  | Receives compressed image stream for detection (via YOLO)              |


## Published Topics

| Topic                  | Message Type                 | Purpose                                                                                       |
|------------------------|------------------------------|-----------------------------------------------------------------------------------------------|
| `car_on_crosswalk`     | `std_msgs/Bool`              | `True` = crosswalk is clear, `False` = car detected on zebra (published every frame)         |
| `traffic_light_state`  | `std_msgs/String`            | `"start"` = green light seen, `"stop"` = red seen or no signal detected — last known state is reused if none is seen |
| `detection_image`      | `sensor_msgs/CompressedImage`| Publishes a JPEG debug image with annotated bounding boxes and masks                          |
