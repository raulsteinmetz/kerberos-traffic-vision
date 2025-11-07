# traffic_detection.py

'''
This node runs 3 yolo v12 models on raw images comming from a ros topic.
In order for it to process images, it needs to be explicitly activated through /flag/detection_switch.
More over, the yolo models should be loaded explicitly using /flag/load_yolo.
The node will publish a debug image, a true flag in case there is no car on the crosswalk, and the
state of the last seen pedestrian traffic light (starting with red for safety).
'''

from pathlib import Path
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory

from .cv_bridge_replacement import rosimg_to_cv2, cv2_to_rosimg_bgr


class DetectionNode(Node):
    def __init__(self):
        super().__init__('traffic_detection')

        # publishers
        self.image_pub = self.create_publisher(Image, 'detection_image', 10) # labeled image
        self.cross_pub = self.create_publisher(Bool, 'car_on_crosswalk', 10) # FALSE if THERE IS a car on the crosswalk
        self.tl_pub = self.create_publisher(String, 'traffic_light_state', 10) # start / stop based on pedestrian traffic light

        # subscribers
        self.create_subscription(String, '/flag/detection_switch', self.signal_cb, 10) # turns node on
        self.create_subscription(Bool, '/flag/load_yolo', self.load_cb, 10) # loads yolo into gpu memory
        self.create_subscription(Image, '/usb_cam1/image_raw', self.image_cb, 10) # images from camera

        # node active?
        self.start = False
        # yolo models
        self.car_model = None
        self.sig_model = None
        self.cross_model = None

        # path of the yolo models (relative to package directory)
        share = Path(get_package_share_directory('traffic_detection')) / 'models'
        self.car_w = share / 'cars.pt'
        self.sig_w = share / 'signals.pt'
        self.cross_w = share / 'cross.pt'

        # parameters for detection
        self.CONF = 0.25
        self.MASK_T = 0.25
        self.D_KERN = 15
        self.IM = 640

        self.last_tl_state = 'stop' # logic starts with red traffic light logic to avoid problems

        self.get_logger().info('Node initialized.')

    # callbacks

    def signal_cb(self, msg: String): # node on/off handler
        self.start = msg.data.strip().lower() == 'start'
        self.get_logger().info('Received "start" flag and activated the node.')

    def load_cb(self, msg: Bool): # load yolo models when reading a 'true' message, unloading with 'false'
        if msg.data:
            try:
                self.car_model = YOLO(str(self.car_w))
                self.sig_model = YOLO(str(self.sig_w))
                self.cross_model = YOLO(str(self.cross_w))
                self.get_logger().info('YOLO models loaded successfully')
            except Exception as e:
                self.get_logger().error(f'CRITICAL ERROR - Model load failed: {e}')
        else:
            if self.car_model == None:
                self.get_logger().info('Unload models flag received, but models were not loaded.')
            self.car_model = self.sig_model = self.cross_model = None
            self.get_logger().info('YOLO models unloaded.')

    def image_cb(self, msg: Image): # processes images, publishes topics
        if not (self.start and self.car_model and self.sig_model and self.cross_model):
            return

        # convert ros images
        try:
            img = rosimg_to_cv2(msg)
        except Exception as e:
            self.get_logger().warn(f'CRITICAL ERROR - Image decoding failed: {e}')
            return

        if img is None:
            return

        img = cv2.resize(img, (self.IM, self.IM)) # resize image to yolo input shape
        vis = img.copy()

        # get predictions from yolo models
        car_boxes, car_lbls = self.detect(self.car_model, img)
        sig_boxes, sig_lbls = self.detect(self.sig_model, img)
        cross_res = self.cross_model.predict(img, imgsz=self.IM, conf=self.CONF, verbose=False)[0]

        # expanding cross walk masks (making sure it invades nearby car bounding boxes)
        x_union = np.zeros(vis.shape[:2], bool)
        if cross_res.masks is not None:
            keep = cross_res.boxes.conf.detach().cpu().numpy() >= self.CONF
            masks = cross_res.masks.data.detach().cpu().numpy()[keep]
            uni = np.any(masks > self.MASK_T, 0).astype(np.uint8)
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.D_KERN, self.D_KERN))
            x_union = cv2.dilate(uni, kern, 1).astype(bool)
            vis = self.overlay_masks(masks, vis, (0, 255, 0))

        # check for car on top of crosswalk through mask and bounding box collision
        car_on = False
        h, w = vis.shape[:2]
        for (x1, y1, x2, y2), lbl in zip(car_boxes, car_lbls):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            y1c, y2c = max(0, y1), min(h, y2)
            x1c, x2c = max(0, x1), min(w, x2)
            if y1c < y2c and x1c < x2c and np.any(x_union[y1c:y2c, x1c:x2c]):
                car_on = True
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vis, lbl, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # check for pedestrian traffic light state
        red_detected = any('red' in lbl.lower() for lbl in sig_lbls) # in case models are changed and labels are slightly different
        green_detected = any('green' in lbl.lower() for lbl in sig_lbls)
        if red_detected: # red has priority in case both are detected
            self.last_tl_state = 'stop'
        elif green_detected:
            self.last_tl_state = 'start'


        # creating debug image
        for (x1, y1, x2, y2), lbl in zip(sig_boxes, sig_lbls):
            col = (0, 0, 255) if 'r' in lbl.lower() else (0, 255, 0)
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            cv2.putText(vis, lbl, (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        # overlay: car-on-crosswalk status
        txt = f'Car in crosswalk: {car_on}'
        col = (0, 255, 255) if car_on else (255, 0, 0)
        cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

        # overlay: current traffic light state (last_tl_state)
        tl_txt = f'TL: {self.last_tl_state.upper()}'
        tl_col = (0, 0, 255) if self.last_tl_state == 'stop' else (0, 255, 0)
        cv2.putText(vis, tl_txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tl_col, 2)

        # publish data to topics
        self.cross_pub.publish(Bool(data=not car_on))  # NOT car_on
        self.tl_pub.publish(String(data=self.last_tl_state))

        dbg_msg = cv2_to_rosimg_bgr(vis, self.get_clock().now().to_msg(), frame_id='detector')
        self.image_pub.publish(dbg_msg)

    # helper for running yolo detection
    def detect(self, model, img):
        res = model.predict(img, imgsz=self.IM, conf=self.CONF, verbose=False)[0]
        boxes, lbls = [], []
        if res.boxes is None:
            return boxes, lbls
        for box, cls in zip(res.boxes.xyxy, res.boxes.cls):
            boxes.append(tuple(map(float, box.tolist())))
            lbls.append(model.names[int(cls)])
        return boxes, lbls

    # helper for overlaying masks on the image
    @staticmethod
    def overlay_masks(masks, img, color):
        for m in masks:
            m = m > 0.5
            col = np.zeros_like(img, np.uint8)
            col[:] = color
            img = np.where(m[..., None], cv2.addWeighted(img, 0.6, col, 0.4, 0), img)
        return img


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
