#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Bool
from sensor_msgs.msg import CompressedImage

import cv2
import os
import numpy as np


class TestController(Node):
    def __init__(self):
        super().__init__('test_controller')

        self.detection_switch_pub = self.create_publisher(String, '/flag/detection_switch', 10)
        self.load_yolo_pub = self.create_publisher(Bool, '/flag/load_yolo', 10)
        self.image_pub = self.create_publisher(CompressedImage, '/usb_cam1/image_raw/compressed', 10)

        self.create_subscription(CompressedImage, 'detection_image', self.image_callback, 10)
        self.create_subscription(String, 'traffic_light_state', self.traffic_cb, 10)
        self.create_subscription(Bool, 'car_on_crosswalk', self.crosswalk_cb, 10)

        image_dir = os.path.expanduser('/home/raul/ros2_humble/src/traffic-detection/test-data')
        self.image_files = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith('.png')
        ])
        self.image_index = 0

        self.traffic_state = "unknown"
        self.crosswalk_state = None

        self.timer = self.create_timer(.5, self.timer_callback)
        self.state = 0

        self.get_logger().info(f'TestController started. {len(self.image_files)} images loaded.')

    def traffic_cb(self, msg: String):
        self.traffic_state = msg.data

    def crosswalk_cb(self, msg: Bool):
        self.crosswalk_state = msg.data

    def timer_callback(self):
        if self.state == 0:
            self.detection_switch_pub.publish(String(data='Start'))
            self.get_logger().info('Published: Start')
        elif self.state == 1:
            self.load_yolo_pub.publish(Bool(data=True))
            self.get_logger().info('Published: load_yolo = True')

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
                self.get_logger().info(f'Traffic light state: {self.traffic_state}, Car on crosswalk, safe to proceed?: {self.crosswalk_state}')

            self.image_index += 1
        else:
            self.get_logger().info('All images published. Sending Stop signal.')
            self.detection_switch_pub.publish(String(data='Stop'))
            self.load_yolo_pub.publish(Bool(data=False))
            self.get_logger().info('Published stop and unload signals.')
            self.timer.cancel()

        self.state += 1

    def image_callback(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv2.imshow("YOLO Detection Debug", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error decoding compressed image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = TestController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
