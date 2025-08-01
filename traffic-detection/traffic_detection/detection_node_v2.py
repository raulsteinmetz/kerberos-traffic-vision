#!/usr/bin/env python3
import os
import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import String, Bool
from sensor_msgs.msg import CompressedImage, Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

from ultralytics import YOLO


class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')

        # Publishers
        self.image_pub           = self.create_publisher(Image,        'detection_image', 10)
        self.car_on_xwalk_pub    = self.create_publisher(Bool,         'car_on_crosswalk', 10)
        self.tl_state_pub        = self.create_publisher(String,       'traffic_light_state', 10)
        self.marker_pub          = self.create_publisher(MarkerArray,  'marker_array',     10)

        # Subscribers
        self.create_subscription(String,        '/flag/detection_switch', self.signal_cb,    10)
        self.create_subscription(Bool,          '/flag/load_yolo',        self.load_cb,      10)
        self.create_subscription(CompressedImage,'/usb_cam1/image_raw/compressed',
                                 self.image_cb, 10)

        # Internal state
        self.start_flag    = False
        self.device        = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.car_seg       = None
        self.xwalk_seg     = None
        self.tl_model      = None
        self.green_count   = 0

        share_dir = get_package_share_directory('traffic_detection')
        self.seg_dir = os.path.join(share_dir, 'models', 'seg')
        self.bbox_dir= os.path.join(share_dir, 'models', 'bb')

        self.bridge = CvBridge()
        self.get_logger().info('DetectionNode ready.')

    # --- control callbacks --------------------------------
    def signal_cb(self, msg: String):
        self.start_flag = (msg.data == 'Start')
        self.get_logger().info(f'Detection {"STARTED" if self.start_flag else "STOPPED"}')

    def load_cb(self, msg: Bool):
        if msg.data:
            self._load_models()
        else:
            self._unload_models()

    # --- model management ---------------------------------
    def _load_models(self):
        try:
            car_pt    = os.path.join(self.seg_dir,   'car.pt')
            xwalk_pt  = os.path.join(self.seg_dir,   'crosswalk.pt')
            tl_pt     = os.path.join(self.bbox_dir, 'lights.pt')

            self.car_seg   = YOLO(car_pt).to(self.device)
            self.xwalk_seg = YOLO(xwalk_pt).to(self.device)
            self.tl_model  = YOLO(tl_pt).to(self.device)

            self.get_logger().info('Segmentation & bbox models loaded.')
        except Exception as e:
            self.get_logger().error(f'Model load failed: {e}')

    def _unload_models(self):
        for attr in ('car_seg','xwalk_seg','tl_model'):
            mdl = getattr(self, attr)
            if mdl:
                del mdl
        torch.cuda.empty_cache()
        self.car_seg = self.xwalk_seg = self.tl_model = None
        self.get_logger().info('Models unloaded.')

    # --- image callback -----------------------------------
    def image_cb(self, msg: CompressedImage):
        if not self.start_flag or self.car_seg is None or self.xwalk_seg is None or self.tl_model is None:
            return

        # decode
        frame = self.bridge.compressed_imgmsg_to_cv2(msg)
        frame = cv2.resize(frame, (640, 640))
        h, w = frame.shape[:2]

        # 1) segmentation -> masks
        car_res   = self.car_seg.predict(source=frame, imgsz=(h,w), conf=0.2)[0]
        xw_res    = self.xwalk_seg.predict(source=frame, imgsz=(h,w), conf=0.2)[0]

        overlap = self._compute_overlap(car_res, xw_res)

        # publish car_on_crosswalk:
        # True = safe to go (no overlap), False = car detected on zebra
        safe = not overlap
        self.car_on_xwalk_pub.publish(Bool(data=safe))

        # 2) bbox detection for traffic lights
        tl_res = self.tl_model.predict(source=frame, imgsz=(h,w), conf=0.3)[0]
        tl_boxes, tl_classes = self._extract_tl(tl_res)

        # update green_count
        if any(c=='G_Signal' for c in tl_classes):
            self.green_count += 1
        else:
            self.green_count = 0

        state = 'start' if self.green_count >= 20 else 'stop'
        self.tl_state_pub.publish(String(data=state))

        # 3) draw debug image
        vis = frame.copy()
        self._overlay_masks(vis, car_res,   (0,0,255))
        self._overlay_masks(vis, xw_res,    (0,255,0))
        self._draw_bboxes(vis, tl_res, {
            'R_Signal':(0,0,255),
            'G_Signal':(0,255,0),
        })

        self.publish_debug_image(vis)

        # 4) publish marker_array
        self._publish_markers(msg.header.frame_id, car_res, tl_res)

    # --- helpers ------------------------------------------
    def _compute_overlap(self, car_res, xw_res):
        if car_res.masks is None or xw_res.masks is None:
            return False
        car_masks = (car_res.masks.data.cpu().numpy()   > 0.2)
        xw_masks  = (xw_res.masks.data.cpu().numpy()    > 0.2)
        car_u     = np.any(car_masks, axis=0)
        xw_u      = np.any(xw_masks,  axis=0)
        intersection = car_u & xw_u
        return bool(intersection.any())

    def _overlay_masks(self, img, res, color, alpha=0.5):
        if res.masks is None: return
        masks = res.masks.data.cpu().numpy() > 0.2
        for m in masks:
            m = m.astype(np.uint8)
            colored = np.zeros_like(img, dtype=np.uint8)
            colored[:] = color
            img[:] = np.where(
                m[...,None],
                cv2.addWeighted(img,1-alpha,colored,alpha,0),
                img
            )

    def _extract_tl(self, res):
        boxes, classes = [], []
        for *xyxy, conf, cls in res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy():
            # unpack from ultralytics Result
            x1,y1,x2,y2 = map(int, xyxy)
            name = res.names[int(cls)]
            if name in ('R_Signal','G_Signal') and conf >= 0.3:
                boxes.append((x1,y1,x2,y2))
                classes.append(name)
        return boxes, classes

    def _draw_bboxes(self, img, res, color_map):
        for box,cls in zip(*self._extract_tl(res)):
            x1,y1,x2,y2 = box
            clr = color_map.get(cls,(255,255,255))
            cv2.rectangle(img,(x1,y1),(x2,y2),clr,2)
            cv2.putText(img,cls,(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,clr,2)

    def publish_debug_image(self, img):
        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        self.image_pub.publish(msg)

    def _publish_markers(self, frame_id, car_res, tl_res):
        ma = MarkerArray()
        idx = 0

        # helper to push one marker per box center
        def add_markers(res, color, ns):
            nonlocal idx
            for *xyxy, _, cls in zip(
                    res.boxes.xyxy.cpu().numpy(),
                    res.boxes.conf.cpu().numpy(),
                    res.boxes.cls.cpu().numpy()):
                x1,y1,x2,y2 = map(int, xyxy)
                cx, cy = (x1+x2)/2, (y1+y2)/2
                m = Marker()
                m.header.frame_id = frame_id
                m.header.stamp    = self.get_clock().now().to_msg()
                m.ns             = ns
                m.id             = idx
                idx += 1
                m.type           = Marker.SPHERE
                m.action         = Marker.ADD
                # place in 2D image plane at z=0
                m.pose.position = Point(x=cx*0.01, y=cy*0.01, z=0.0)
                m.pose.orientation.w = 1.0
                # sphere size
                m.scale.x = m.scale.y = m.scale.z = 0.05
                # color
                m.color.r, m.color.g, m.color.b = color
                m.color.a = 1.0
                ma.markers.append(m)

        # cars = red, green for G_Signal, etc.
        add_markers(car_res, (1.0,0.0,0.0), 'cars')
        add_markers(tl_res,  (0.0,1.0,0.0), 'traffic_lights')

        self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    finally:
        node._unload_models()
        node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    main()
