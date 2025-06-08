#!/usr/bin/env python3
import os
import cv2
import torch

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import String, Bool
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError


class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')

        # --- publishers ---
        self.image_pub = self.create_publisher(Image, 'detection_image', 10)

        # --- subscribers ---
        self.create_subscription(String, '/flag/detection_switch',
                                 self.signal_callback, 10)
        self.create_subscription(Bool, '/flag/load_yolo',
                                 self.load_yolo_callback, 10)
        self.create_subscription(CompressedImage,
                                 '/usb_cam1/image_raw/compressed',
                                 self.image_callback, 10)

        # --- state logic ---
        self.start_flag = False
        self.car_model = None
        self.cwtl_model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # locate models inside the installed package
        share_dir = get_package_share_directory('traffic_detection')
        self.models_dir = os.path.join(share_dir, 'models')

        self.bridge = CvBridge()
        self.get_logger().info('DetectionNode ready (debug-image mode)')

    # ---  control callbacks  ---
    def signal_callback(self, msg: String):
        self.start_flag = (msg.data == 'Start')
        self.get_logger().info(f'Detection {"started" if self.start_flag else "stopped"}.')

    def load_yolo_callback(self, msg: Bool):
        self.load_models() if msg.data else self.unload_models()

    # ---  model management  ---
    def load_models(self):
        try:
            car_pt  = os.path.join(self.models_dir, 'car_model.pt')
            tl_pt   = os.path.join(self.models_dir, 'crosswalk_trafficlights.pt')
            self.car_model  = torch.hub.load('ultralytics/yolov5', 'custom',
                                             path=car_pt).to(self.device)
            self.cwtl_model = torch.hub.load('ultralytics/yolov5', 'custom',
                                             path=tl_pt).to(self.device)
            self.get_logger().info('YOLO models loaded.')
        except Exception as e:
            self.get_logger().error(f'Model load failed: {e}')

    def unload_models(self):
        try:
            del self.car_model, self.cwtl_model
            torch.cuda.empty_cache()
        except AttributeError:
            pass
        self.car_model = self.cwtl_model = None
        self.get_logger().info('YOLO models unloaded.')

    # ---  image callback  ---
    def image_callback(self, msg: CompressedImage):
        if not self.start_flag or self.car_model is None or self.cwtl_model is None:
            return

        try:
            cv_img = self.bridge.compressed_imgmsg_to_cv2(msg)
        except CvBridgeError as e:
            self.get_logger().error(f'CV-Bridge error: {e}')
            return

        # run both models
        car_boxes, _  = self.get_boxes(self.car_model,  cv_img,
                                       ["car", "motorcycle", "bus", "truck", "train"])
        cw_boxes, _   = self.get_boxes(self.cwtl_model, cv_img, ["Zebra_Cross"])
        tl_boxes, tl_cls = self.get_boxes_with_class(self.cwtl_model, cv_img,
                                                     ["R_Signal", "G_Signal"])

        annotated = self.draw_boxes(cv_img.copy(), car_boxes, cw_boxes,
                                    tl_boxes, tl_cls)

        self.publish_debug_image(annotated)

    # ---  helper functions  ---
    def get_boxes(self, model, img, wanted, conf_thr=0.3):
        res = model(img)
        boxes, scores = [], []
        for *xyxy, conf, cls in res.xyxy[0]:
            if conf >= conf_thr and model.names[int(cls)] in wanted:
                boxes.append(xyxy)
                scores.append(conf.item())
        return boxes, scores

    def get_boxes_with_class(self, model, img, wanted, conf_thr=0.3):
        res = model(img)
        boxes, classes = [], []
        for *xyxy, conf, cls in res.xyxy[0]:
            name = model.names[int(cls)]
            if conf >= conf_thr and name in wanted:
                boxes.append(xyxy)
                classes.append(name)
        return boxes, classes

    def draw_boxes(self, img, car_b, cross_b, tl_b, tl_cls):
        # cars = blue, crosswalks = green, traffic lights red/green
        for b in car_b:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                          (255, 0, 0), 2)
            cv2.putText(img, 'Car', (int(b[0]), int(b[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        for b in cross_b:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                          (0, 255, 0), 2)
            cv2.putText(img, 'Crosswalk', (int(b[0]), int(b[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for b, cls in zip(tl_b, tl_cls):
            color = (0, 0, 255) if cls == "R_Signal" else (0, 255, 0)
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                          color, 2)
            cv2.putText(img, cls, (int(b[0]), int(b[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img

    def publish_debug_image(self, img):
        try:
            img_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            self.image_pub.publish(img_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'Debug image publish error: {e}')


# ---  main ---
def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.unload_models()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
