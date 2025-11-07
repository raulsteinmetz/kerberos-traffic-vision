# test_traffic_detection.py

'''
Tester node for "traffic_detection.py"
Starts by activating the main node and loading yolos through flags.
Loads images from a specified directory and publishes 4 per second for debbuging.
Deactivates main node and unloads yolo models.
'''

import os
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
import numpy as np

from .cv_bridge_replacement import cv2_to_rosimg_bgr

class TestController(Node):
    def __init__(self):
        super().__init__('test_traffic_detection')

        # publishers
        self.detection_switch_pub = self.create_publisher(String, '/flag/detection_switch', 10)
        self.load_yolo_pub = self.create_publisher(Bool, '/flag/load_yolo', 10)
        self.image_pub = self.create_publisher(Image, '/usb_cam1/image_raw', 10)  # RAW Image

        # datasetpath (hardcoded)
        self.image_dir = '/home/raul/ros2_ws/src/traffic-detection/traffic_detection/test_data/dataset_1'
        exts = ('.png', '.jpg', '.jpeg', '.bmp')
        self.image_files = sorted(
            [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.lower().endswith(exts)]
        ) if os.path.exists(self.image_dir) else []

        self.image_index = 0
        self.state = 0

        # publish 4 images per second
        self.timer = self.create_timer(0.25, self.timer_callback)
        self.get_logger().info(f'Test node initialized. {len(self.image_files)} images loaded from {self.image_dir}')

    def timer_callback(self):
        # activate node and load yolo
        if self.state == 0:
            self.detection_switch_pub.publish(String(data='Start'))
            self.get_logger().info('Published: Start for turning node on')
        elif self.state == 1:
            self.load_yolo_pub.publish(Bool(data=True))
            self.get_logger().info('Published: load_yolo = True')

        # publish next image
        if self.image_index < len(self.image_files):
            path = self.image_files[self.image_index]
            img = cv2.imread(path)  # BGR
            if img is None:
                self.get_logger().error(f"CRITICAL ERROR - Failed to load image: {path}")
            else:
                msg = cv2_to_rosimg_bgr(img, self.get_clock().now().to_msg(), frame_id="test_cam")
                self.image_pub.publish(msg)
                self.get_logger().info(f'Published: {os.path.basename(path)}')
            self.image_index += 1
        else:
            self.get_logger().info('All images published. Sending Stop signal and unload models flag.')
            self.detection_switch_pub.publish(String(data='Stop'))
            self.load_yolo_pub.publish(Bool(data=False))
            self.timer.cancel()

        self.state += 1

def main(args=None):
    rclpy.init(args=args)
    node = TestController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
