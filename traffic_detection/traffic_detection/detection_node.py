#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Bool
from sensor_msgs.msg import CompressedImage, Image
from visualization_msgs.msg import MarkerArray

from ament_index_python.packages import get_package_share_directory

import torch
import os


class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')

        # --- publishers ---
        self.car_pub = self.create_publisher(Bool, 'car_on_crosswalk', 10)
        self.traffic_pub = self.create_publisher(String, 'traffic_light_state', 10)
        self.image_pub = self.create_publisher(Image, 'detection_image', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'marker_array', 10)

        # --- subscribers ---
        self.create_subscription(String, '/flag/detection_switch', self.signal_callback, 10)
        self.create_subscription(Bool, '/flag/load_yolo', self.load_yolo_callback, 10)
        self.create_subscription(CompressedImage, '/usb_cam1/image_raw/compressed', self.image_callback, 10)

        # --- state logic ---
        self.start_flag = False
        self.car_model = None
        self.crosswalk_model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # --- locating the .pt models folder ---
        pkg_share_path = get_package_share_directory('traffic_detection')
        self.models_path = os.path.join(pkg_share_path, 'models')

        self.get_logger().info('DetectionNode initialized.')

    def signal_callback(self, msg: String): # detection on-off switch
        if msg.data == 'Start':
            self.start_flag = True
            self.get_logger().info('Detection started.')
        elif msg.data == 'Stop':
            self.start_flag = False
            self.get_logger().info('Detection stopped.')

    def load_yolo_callback(self, msg: Bool): # load or unload torch models switch
        if msg.data:
            self.load_models()
        else:
            self.unload_models()

    def load_models(self): # auxiliary load models function
        try:
            car_model_path = os.path.join(self.models_path, 'car_model.pt')
            crosswalk_model_path = os.path.join(self.models_path, 'crosswalk_trafficlights.pt')

            self.car_model = torch.hub.load('ultralytics/yolov5', 'custom', path=car_model_path).to(self.device)
            self.crosswalk_model = torch.hub.load('ultralytics/yolov5', 'custom', path=crosswalk_model_path).to(self.device)

            self.get_logger().info('YOLO models loaded successfully.')
        except Exception as e:
            self.get_logger().error(f'Failed to load models: {e}')

    def unload_models(self): # auxiliary unload models function
        try:
            if self.car_model is not None:
                del self.car_model
            if self.crosswalk_model is not None:
                del self.crosswalk_model
            torch.cuda.empty_cache()

            self.car_model = None
            self.crosswalk_model = None

            self.get_logger().info('YOLO models unloaded successfully.')
        except Exception as e:
            self.get_logger().error(f'Failed to unload models: {e}')

    def image_callback(self, msg: CompressedImage): # processing images from camera
        if not self.start_flag:
            self.get_logger().info('Detection is paused. Skipping image processing.')
            return
        if self.car_model is None or self.crosswalk_model is None:
            self.get_logger().info('Models not loaded. Skipping image processing.')
            return

        self.get_logger().info('Received image frame. YOLO inference would happen here.')


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
