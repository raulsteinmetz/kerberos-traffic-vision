#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Bool
from sensor_msgs.msg import CompressedImage, Image
from visualization_msgs.msg import MarkerArray


class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')

        # --- --- pubs --- ---
        self.car_pub = self.create_publisher(Bool, 'car_on_crosswalk', 10)
        self.traffic_pub = self.create_publisher(String, 'traffic_light_state', 10)
        self.image_pub = self.create_publisher(Image, 'detection_image', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'marker_array', 10)

        # --- --- subs --- ---
        self.create_subscription(String, '/flag/detection_switch', self.signal_callback, 10)
        self.create_subscription(Bool, '/flag/load_yolo', self.load_yolo_callback, 10)
        self.create_subscription(CompressedImage, '/usb_cam1/image_raw/compressed', self.image_callback, 10)

        self.get_logger().info('DetectionNode started.')

    def signal_callback(self, msg: String):
        self.get_logger().info(f'Received detection switch: {msg.data}')

    def load_yolo_callback(self, msg: Bool):
        self.get_logger().info(f'Received load_yolo flag: {msg.data}')

    def image_callback(self, msg: CompressedImage):
        self.get_logger().info('Received image frame.')


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
