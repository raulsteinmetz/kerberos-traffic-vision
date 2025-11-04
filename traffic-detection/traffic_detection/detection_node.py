# #!/usr/bin/env python3
# import os
# from pathlib import Path

# import cv2
# import numpy as np
# import rclpy
# from ament_index_python.packages import get_package_share_directory
# from rclpy.node import Node
# from sensor_msgs.msg import CompressedImage
# from std_msgs.msg import Bool, String
# from ultralytics import YOLO


# class DetectionNode(Node):
#     def __init__(self):
#         super().__init__("detection_node")

#         self.image_pub = self.create_publisher(CompressedImage, "detection_image", 10)
#         self.cross_pub = self.create_publisher(Bool, "car_on_crosswalk", 10)
#         self.tl_pub = self.create_publisher(String, "traffic_light_state", 10)

#         self.create_subscription(String, "/flag/detection_switch", self.signal_cb, 10)
#         self.create_subscription(Bool, "/flag/load_yolo", self.load_cb, 10)
#         self.create_subscription(
#             CompressedImage,
#             "/usb_cam1/image_raw/compressed",
#             self.image_cb,
#             10,
#         )

#         self.start = False
#         self.car_model = None
#         self.sig_model = None
#         self.cross_model = None

#         share = Path(get_package_share_directory("traffic_detection")) / "models"
#         self.car_w = share / "cars.pt"
#         self.sig_w = share / "signals.pt"
#         self.cross_w = share / "cross.pt"

#         self.CONF = 0.25
#         self.MASK_T = 0.25
#         self.D_KERN = 15
#         self.IM = 640

#         self.last_tl_state = "stop"

#     def signal_cb(self, msg: String):
#         self.start = msg.data.strip().lower() == "start"

#     def load_cb(self, msg: Bool):
#         if msg.data:
#             try:
#                 self.car_model = YOLO(str(self.car_w))
#                 self.sig_model = YOLO(str(self.sig_w))
#                 self.cross_model = YOLO(str(self.cross_w))
#             except Exception as e:
#                 self.get_logger().error(f"model load: {e}")
#         else:
#             self.car_model = self.sig_model = self.cross_model = None

#     def image_cb(self, msg: CompressedImage):
#         if not (self.start and self.car_model and self.sig_model and self.cross_model):
#             return
#         buf = np.frombuffer(msg.data, np.uint8)
#         img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
#         if img is None:
#             return
#         img = cv2.resize(img, (self.IM, self.IM))
#         vis = img.copy()

#         car_boxes, car_lbls = self.detect(self.car_model, img)
#         sig_boxes, sig_lbls = self.detect(self.sig_model, img)

#         cross_res = self.cross_model.predict(img, imgsz=self.IM, conf=self.CONF, verbose=False)[0]
#         x_union = np.zeros(vis.shape[:2], bool)
#         if cross_res.masks is not None:
#             keep = cross_res.boxes.conf.cpu().numpy() >= self.CONF
#             masks = cross_res.masks.data.cpu().numpy()[keep]
#             uni = np.any(masks > self.MASK_T, 0).astype(np.uint8)
#             kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.D_KERN, self.D_KERN))
#             x_union = cv2.dilate(uni, kern, 1).astype(bool)
#             vis = self.overlay_masks(masks, vis, (0, 255, 0))

#         car_on = False
#         h, w = vis.shape[:2]
#         for (x1, y1, x2, y2), lbl in zip(car_boxes, car_lbls):
#             x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
#             if np.any(x_union[y1:y2, x1:x2]):
#                 car_on = True
#             cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             cv2.putText(vis, lbl, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

#         red_detected   = any("red"   in lbl.lower() for lbl in sig_lbls)
#         green_detected = any("green" in lbl.lower() for lbl in sig_lbls)

#         if red_detected:
#             self.last_tl_state = "stop"
#         elif green_detected:
#             self.last_tl_state = "start"

#         for (x1, y1, x2, y2), lbl in zip(sig_boxes, sig_lbls):
#             col = (0, 0, 255) if "r" in lbl.lower() else (0, 255, 0)
#             cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
#             cv2.putText(vis, lbl, (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

#         txt = f"Car in crosswalk: {car_on}"
#         col = (0, 255, 255) if car_on else (255, 0, 0)
#         cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

#         self.cross_pub.publish(Bool(data=not car_on))
#         self.tl_pub.publish(String(data=self.last_tl_state))

#         self.pub_img(vis)

#     def detect(self, model, img):
#         res = model.predict(img, imgsz=self.IM, conf=self.CONF, verbose=False)[0]
#         boxes, lbls = [], []
#         if res.boxes is None:
#             return boxes, lbls
#         for box, cls in zip(res.boxes.xyxy, res.boxes.cls):
#             boxes.append(tuple(box.tolist()))
#             lbls.append(model.names[int(cls)])
#         return boxes, lbls

#     @staticmethod
#     def overlay_masks(masks, img, color):
#         for m in masks:
#             m = m > 0.5
#             col = np.zeros_like(img, np.uint8)
#             col[:] = color
#             img = np.where(m[..., None], cv2.addWeighted(img, 0.6, col, 0.4, 0), img)
#         return img

#     def pub_img(self, img):
#         ok, buf = cv2.imencode(".jpg", img)
#         if not ok:
#             return
#         msg = CompressedImage()
#         msg.format = "jpeg"
#         msg.data = buf.tobytes()
#         self.image_pub.publish(msg)


# def main(args=None):
#     rclpy.init(args=args)
#     node = DetectionNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from ultralytics import YOLO
from cv_bridge import CvBridge


class DetectionNode(Node):
    def __init__(self):
        super().__init__("detection_node")

        self.image_pub = self.create_publisher(Image, "detection_image", 10)
        self.cross_pub = self.create_publisher(Bool, "car_on_crosswalk", 10)
        self.tl_pub = self.create_publisher(String, "traffic_light_state", 10)

        self.create_subscription(String, "/flag/detection_switch", self.signal_cb, 10)
        self.create_subscription(Bool, "/flag/load_yolo", self.load_cb, 10)
        self.create_subscription(
            Image,
            "/usb_cam1/image_raw",
            self.image_cb,
            10,
        )

        self.start = False
        self.car_model = None
        self.sig_model = None
        self.cross_model = None

        share = Path(get_package_share_directory("traffic_detection")) / "models"
        self.car_w = share / "cars.pt"
        self.sig_w = share / "signals.pt"
        self.cross_w = share / "cross.pt"

        self.CONF = 0.25
        self.MASK_T = 0.25
        self.D_KERN = 15
        self.IM = 640

        self.last_tl_state = "stop"

        self.bridge = CvBridge()

    def signal_cb(self, msg: String):
        self.start = msg.data.strip().lower() == "start"

    def load_cb(self, msg: Bool):
        if msg.data:
            try:
                self.car_model = YOLO(str(self.car_w))
                self.sig_model = YOLO(str(self.sig_w))
                self.cross_model = YOLO(str(self.cross_w))
            except Exception as e:
                self.get_logger().error(f"model load: {e}")
        else:
            self.car_model = self.sig_model = self.cross_model = None

    def image_cb(self, msg: Image):
        if not (self.start and self.car_model and self.sig_model and self.cross_model):
            return

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if img is None:
            return
        img = cv2.resize(img, (self.IM, self.IM))
        vis = img.copy()

        car_boxes, car_lbls = self.detect(self.car_model, img)
        sig_boxes, sig_lbls = self.detect(self.sig_model, img)

        cross_res = self.cross_model.predict(img, imgsz=self.IM, conf=self.CONF, verbose=False)[0]
        x_union = np.zeros(vis.shape[:2], bool)
        if cross_res.masks is not None:
            keep = cross_res.boxes.conf.cpu().numpy() >= self.CONF
            masks = cross_res.masks.data.cpu().numpy()[keep]
            uni = np.any(masks > self.MASK_T, 0).astype(np.uint8)
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.D_KERN, self.D_KERN))
            x_union = cv2.dilate(uni, kern, 1).astype(bool)
            vis = self.overlay_masks(masks, vis, (0, 255, 0))

        car_on = False
        h, w = vis.shape[:2]
        for (x1, y1, x2, y2), lbl in zip(car_boxes, car_lbls):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            if np.any(x_union[y1:y2, x1:x2]):
                car_on = True
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vis, lbl, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        red_detected   = any("red"   in lbl.lower() for lbl in sig_lbls)
        green_detected = any("green" in lbl.lower() for lbl in sig_lbls)

        if red_detected:
            self.last_tl_state = "stop"
        elif green_detected:
            self.last_tl_state = "start"

        for (x1, y1, x2, y2), lbl in zip(sig_boxes, sig_lbls):
            col = (0, 0, 255) if "r" in lbl.lower() else (0, 255, 0)
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            cv2.putText(vis, lbl, (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        txt = f"Car in crosswalk: {car_on}"
        col = (0, 255, 255) if car_on else (255, 0, 0)
        cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

        self.cross_pub.publish(Bool(data=not car_on))
        self.tl_pub.publish(String(data=self.last_tl_state))

        self.pub_img(vis)

    def detect(self, model, img):
        res = model.predict(img, imgsz=self.IM, conf=self.CONF, verbose=False)[0]
        boxes, lbls = [], []
        if res.boxes is None:
            return boxes, lbls
        for box, cls in zip(res.boxes.xyxy, res.boxes.cls):
            boxes.append(tuple(box.tolist()))
            lbls.append(model.names[int(cls)])
        return boxes, lbls

    @staticmethod
    def overlay_masks(masks, img, color):
        for m in masks:
            m = m > 0.5
            col = np.zeros_like(img, np.uint8)
            col[:] = color
            img = np.where(m[..., None], cv2.addWeighted(img, 0.6, col, 0.4, 0), img)
        return img

    def pub_img(self, img):
        msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self.image_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
