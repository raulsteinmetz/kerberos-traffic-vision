#!/usr/bin/env python3
import os
import cv2
import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Bool
from sensor_msgs.msg import CompressedImage, Image
from visualization_msgs.msg import MarkerArray
from cv_bridge import CvBridge


class TestController(Node):
    def __init__(self):
        super().__init__('test_controller')

        # Publishers
        self.load_pub      = self.create_publisher(Bool,   '/flag/load_yolo',         10)
        self.switch_pub    = self.create_publisher(String, '/flag/detection_switch',  10)
        self.image_pub     = self.create_publisher(CompressedImage,
                                                    '/usb_cam1/image_raw/compressed', 10)

        # Subscribers for all outputs
        self.create_subscription(Bool,        'car_on_crosswalk',   self.cb_car_on_xwalk,  10)
        self.create_subscription(String,      'traffic_light_state',self.cb_tl_state,      10)
        self.create_subscription(Image,       'detection_image',    self.cb_debug_image,    10)
        self.create_subscription(MarkerArray,'marker_array',       self.cb_marker_array,   10)

        # Image loader
        image_dir = os.path.expanduser(
            '~/ros2_humble/src/traffic_detection/test_data/'
        )
        self.images = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith('.png')
        ])
        self.idx = 0

        self.bridge = CvBridge()
        self.get_logger().info(f'Loaded {len(self.images)} test images.')

        # State machine:
        # 0: send load_models
        # 1: send start_detection
        # 2..N+1: send images
        # N+2: send stop + unload, then cancel
        self.state = 0

        # Timer at 0.25s intervals
        self.timer = self.create_timer(0.25, self.timer_cb)

    def timer_cb(self):
        if self.state == 0:
            # Load models into memory
            self.load_pub.publish(Bool(data=True))
            self.get_logger().info('[PUB] /flag/load_yolo = True')
        elif self.state == 1:
            # Start detection
            self.switch_pub.publish(String(data='Start'))
            self.get_logger().info('[PUB] /flag/detection_switch = Start')
        elif 2 <= self.state < 2 + len(self.images):
            # Publish next image
            img_path = self.images[self.idx]
            img = cv2.imread(img_path)
            if img is None:
                self.get_logger().warn(f'Could not load {img_path}')
            else:
                # compress and publish
                frame = cv2.imencode('.jpg', img)[1].tobytes()
                msg = CompressedImage()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.format = 'jpeg'
                msg.data = frame
                self.image_pub.publish(msg)
                self.get_logger().info(f'[PUB] Image: {os.path.basename(img_path)}')

            self.idx += 1

        elif self.state == 2 + len(self.images):
            # Stop detection & unload
            self.switch_pub.publish(String(data='Stop'))
            self.load_pub.publish(Bool(data=False))
            self.get_logger().info('[PUB] /flag/detection_switch = Stop')
            self.get_logger().info('[PUB] /flag/load_yolo = False')
            # done
            self.timer.cancel()
        self.state += 1

    # --- Subscribers callbacks --------------------------------
    def cb_car_on_xwalk(self, msg: Bool):
        self.get_logger().info(f'[SUB] car_on_crosswalk → safe_to_go={msg.data}')

    def cb_tl_state(self, msg: String):
        self.get_logger().info(f'[SUB] traffic_light_state → {msg.data}')

    def cb_debug_image(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imshow('Debug Detection', img)
            cv2.waitKey(1)  # non-blocking
        except Exception as e:
            self.get_logger().error(f'Error decoding debug image: {e}')

    def cb_marker_array(self, msg: MarkerArray):
        n = len(msg.markers)
        self.get_logger().info(f'[SUB] marker_array → {n} markers')

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
