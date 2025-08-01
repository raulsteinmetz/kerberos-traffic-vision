#!/usr/bin/env python3

import rclpy
import os
import cv2
import numpy as np

from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import String, Bool
from sensor_msgs.msg import CompressedImage

from ultralytics import YOLO


class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')

        self.image_pub = self.create_publisher(CompressedImage, 'detection_image', 10)

        self.create_subscription(String, '/flag/detection_switch', self.signal_callback, 10)
        self.create_subscription(Bool, '/flag/load_yolo', self.load_yolo_callback, 10)
        self.create_subscription(CompressedImage, '/usb_cam1/image_raw/compressed', self.image_callback, 10)

        self.start_flag = False
        self.car_model = None
        self.tl_model = None
        self.crosswalk_model = None

        share_dir = get_package_share_directory('traffic_detection')
        self.models_dir = os.path.join(share_dir, 'models')

        self.MIN_CONF = 0.40
        self.MASK_THRESH = 0.25
        self.DILATE_KERNEL = 15
        self.IMG_SIZE = (640, 640)

    def signal_callback(self, msg: String):
        self.start_flag = (msg.data == 'Start')
        self.get_logger().info(f'Detection {"started" if self.start_flag else "stopped"}.')

    def load_yolo_callback(self, msg: Bool):
        self.load_models() if msg.data else self.unload_models()

    def load_models(self):
        try:
            self.car_model = YOLO(os.path.join(self.models_dir, 'cars_bb.pt'))
            self.tl_model = YOLO(os.path.join(self.models_dir, 'lights_bb.pt'))
            self.crosswalk_model = YOLO(os.path.join(self.models_dir, 'crosswalk_seg.pt'))
            self.get_logger().info('All YOLOv11 models loaded.')
        except Exception as e:
            self.get_logger().error(f'Model load failed: {e}')

    def unload_models(self):
        try:
            del self.car_model, self.tl_model, self.crosswalk_model
        except AttributeError:
            pass
        self.car_model = self.tl_model = self.crosswalk_model = None
        self.get_logger().info('Models unloaded.')

    def image_callback(self, msg: CompressedImage):
        if not self.start_flag or not all([self.car_model, self.tl_model, self.crosswalk_model]):
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f'Image decode error: {e}')
            return

        img_resized = cv2.resize(cv_img, self.IMG_SIZE)
        vis = img_resized.copy()

        car_boxes, _, _ = self.detect_objects(self.car_model, img_resized,
                                              ["car", "motorcycle", "bus", "truck", "train"])
        tl_boxes, _, tl_classes = self.detect_objects(self.tl_model, img_resized,
                                                      ["R_Signal", "G_Signal"])

        xwalk_res = self.crosswalk_model.predict(source=img_resized, imgsz=self.IMG_SIZE,
                                                 conf=self.MIN_CONF)[0]

        xwalk_dilated = np.zeros(vis.shape[:2], dtype=bool)
        if getattr(xwalk_res, "masks", None) and getattr(xwalk_res, "boxes", None):
            keep = xwalk_res.boxes.conf.cpu().numpy() >= self.MIN_CONF
            masks = xwalk_res.masks.data.cpu().numpy()[keep]
            xwalk_union = np.any(masks > self.MASK_THRESH, axis=0).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (self.DILATE_KERNEL, self.DILATE_KERNEL))
            xwalk_dilated = cv2.dilate(xwalk_union, kernel, iterations=1).astype(bool)
            xwalk_res.masks.data = xwalk_res.masks.data[keep]
            vis = self.overlay_masks(xwalk_res, vis, (0, 255, 0))

        car_on_cross = False
        h, w = xwalk_dilated.shape
        for (x1, y1, x2, y2) in car_boxes:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
            if np.any(xwalk_dilated[y1:y2, x1:x2]):
                car_on_cross = True
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vis, 'Car', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        for b, cls in zip(tl_boxes, tl_classes):
            x1, y1, x2, y2 = map(int, b)
            color = (0, 0, 255) if cls == "R_Signal" else (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, cls, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        text_color = (0, 255, 255) if car_on_cross else (255, 0, 0)
        cv2.putText(vis, f'Car in crosswalk: {car_on_cross}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        self.publish_debug_image(vis)

    def detect_objects(self, model, img, wanted, conf_thr=0.3):
        res = model.predict(source=img, imgsz=self.IMG_SIZE, conf=conf_thr)[0]
        boxes, scores, classes = [], [], []
        names = model.model.names

        if not hasattr(res, 'boxes') or res.boxes is None:
            return boxes, scores, classes

        for box, cls_id, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
            name = names[int(cls_id)]
            if conf >= conf_thr and name in wanted:
                x1, y1, x2, y2 = box.tolist()
                boxes.append((x1, y1, x2, y2))
                scores.append(conf.item())
                classes.append(name)

        return boxes, scores, classes

    def overlay_masks(self, result, img, color):
        if not getattr(result, "masks", None):
            return img
        masks = result.masks.data.cpu().numpy()
        for m in masks:
            bin_mask = m > self.MASK_THRESH
            colored = np.zeros_like(img, dtype=np.uint8)
            colored[:] = color
            img = np.where(bin_mask[..., None],
                           cv2.addWeighted(img, 0.6, colored, 0.4, 0),
                           img)
        return img

    def publish_debug_image(self, img):
        try:
            _, buffer = cv2.imencode('.jpg', img)
            msg = CompressedImage()
            msg.format = "jpeg"
            msg.data = buffer.tobytes()
            self.image_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Debug image encode/publish error: {e}')


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
