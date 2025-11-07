# cv_bridge_replacement.py
'''
This is a helper defining two cv_bridge functions needed for the functioning of the node.
The kerberos system was having errors with cv_bridge when executing this node, because an
older version of numpy is required for the package, causing imcompatibility issues.
'''

import numpy as np
import cv2
from sensor_msgs.msg import Image

# map for converting ros2 image types
_ENC2DTYPE = {
    "mono8":  (np.uint8, 1),
    "bgr8":   (np.uint8, 3),
    "rgb8":   (np.uint8, 3),
    "rgba8":  (np.uint8, 4),
    "bgra8":  (np.uint8, 4),
}

def rosimg_to_cv2(msg: Image) -> np.ndarray:
    if msg.encoding not in _ENC2DTYPE:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")
    dtype, ch = _ENC2DTYPE[msg.encoding]
    arr = np.frombuffer(msg.data, dtype=dtype)
    if ch == 1:
        img = arr.reshape(msg.height, msg.width)
    else:
        img = arr.reshape(msg.height, msg.width, ch)
    if msg.encoding == "rgb8":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif msg.encoding == "rgba8":
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif msg.encoding == "bgra8":
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(img)

def cv2_to_rosimg_bgr(img_bgr: np.ndarray, stamp, frame_id: str = "") -> Image:
    if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3 or img_bgr.dtype != np.uint8:
        raise ValueError("img_bgr must be uint8 HxWx3")
    h, w, _ = img_bgr.shape
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = h
    msg.width = w
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = w * 3
    msg.data = np.ascontiguousarray(img_bgr).tobytes()
    return msg
