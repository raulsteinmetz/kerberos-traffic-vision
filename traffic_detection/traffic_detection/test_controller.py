#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Bool
from sensor_msgs.msg import CompressedImage

import cv2
import os


class TestController(Node):
    def __init__(self):
        super().__init__('test_controller')

        # --- --- pubs --- ---
        self.detection_switch_pub = self.create_publisher(String, '/flag/detection_switch', 10)
        self.load_yolo_pub = self.create_publisher(Bool, '/flag/load_yolo', 10)
        self.image_pub = self.create_publisher(CompressedImage, '/usb_cam1/image_raw/compressed', 10)

        # --- --- sending images for detection --- ---
        image_dir = os.path.expanduser('~/ros2_ws/src/traffic_detection/test_data/dataset_1')
        self.image_files = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith('.png')
        ])
        self.image_index = 0

        # periodic publishing
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.state = 0

        self.get_logger().info(f'TestController started. {len(self.image_files)} images loaded.')

    def timer_callback(self):
        # 'switch yolo on' and 'load yolo' messages
        if self.state == 0:
            self.detection_switch_pub.publish(String(data='Start'))
            self.get_logger().info('Published: Start')
        elif self.state == 1:
            self.load_yolo_pub.publish(Bool(data=True))
            self.get_logger().info('Published: load_yolo = True')

        # publish one image per call_back
        if self.image_index < len(self.image_files):
            img_path = self.image_files[self.image_index]
            img = cv2.imread(img_path)

            if img is None:
                self.get_logger().warn(f"Failed to load image: {img_path}")
            else:
                msg = CompressedImage()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.format = "jpeg"
                msg.data = list(cv2.imencode('.jpg', img)[1].tobytes())
                self.image_pub.publish(msg)
                self.get_logger().info(f'Published image: {os.path.basename(img_path)}')

            self.image_index += 1
        else:
            self.get_logger().info('All images published. Sending Stop signal.')
            self.detection_switch_pub.publish(String(data='Stop')) # node should stop detection
            self.load_yolo_pub.publish(Bool(data=False)) # free up gpu or cpu memory (delete yolo model)
            self.get_logger().info('Published: Stop and load_yolo = False')
            self.timer.cancel()

        self.state += 1


def main(args=None):
    rclpy.init(args=args)
    node = TestController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
