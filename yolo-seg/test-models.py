#!/usr/bin/env python3

import os
import cv2
import numpy as np
from ultralytics import YOLO

CAR_MODEL_PATH       = "./car_bb.pt"
CROSSWALK_MODEL_PATH = "./crosswalk_seg.pt"
IMAGE_DIR            = "../traffic-detection/test-data/"

WIDTH, HEIGHT  = 640, 640
MIN_CONF       = 0.40
MASK_THRESH    = 0.25
DILATE_KERNEL  = 15

CAR_COLOR        = (0, 0, 255)
XWALK_COLOR      = (0, 255, 0)
OVERLAP_COLOR    = (0, 255, 255)
NO_OVERLAP_COLOR = (255, 0, 0)
ALPHA            = 0.4

car_model   = YOLO(CAR_MODEL_PATH)
xwalk_model = YOLO(CROSSWALK_MODEL_PATH)

def overlay_masks(result, img, color):
    '''Overlay segmentation masks from the result onto an image using a specified color'''
    if not getattr(result, "masks", None):
        return img
    masks = result.masks.data.cpu().numpy()
    for m in masks:
        bin_mask = m > MASK_THRESH
        colored = np.zeros_like(img, dtype=np.uint8)
        colored[:] = color
        img = np.where(bin_mask[..., None],
                       cv2.addWeighted(img, 1 - ALPHA, colored, ALPHA, 0),
                       img)
    return img

for fname in sorted(os.listdir(IMAGE_DIR)):
    path = os.path.join(IMAGE_DIR, fname)
    img = cv2.imread(path)
    if img is None:
        continue

    img = cv2.resize(img, (WIDTH, HEIGHT))
    vis = img.copy()

    car_res   = car_model.predict(source=img, imgsz=(WIDTH, HEIGHT), conf=MIN_CONF)[0]
    xwalk_res = xwalk_model.predict(source=img, imgsz=(WIDTH, HEIGHT), conf=MIN_CONF)[0]

    car_boxes = np.empty((0, 4))
    if getattr(car_res, "boxes", None):
        conf_mask = car_res.boxes.conf.cpu().numpy() >= MIN_CONF
        car_boxes = car_res.boxes.xyxy.cpu().numpy()[conf_mask]
        for (x1, y1, x2, y2) in car_boxes.astype(int):
            cv2.rectangle(vis, (x1, y1), (x2, y2), CAR_COLOR, 2)

    if getattr(xwalk_res, "masks", None) and getattr(xwalk_res, "boxes", None):
        keep = xwalk_res.boxes.conf.cpu().numpy() >= MIN_CONF
        masks = xwalk_res.masks.data.cpu().numpy()[keep]
        xwalk_union = np.any(masks > MASK_THRESH, axis=0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KERNEL, DILATE_KERNEL))
        xwalk_dilated = cv2.dilate(xwalk_union, kernel, iterations=1).astype(bool)
        xwalk_vis_res = xwalk_res
        xwalk_vis_res.masks.data = xwalk_res.masks.data[keep]
        vis = overlay_masks(xwalk_vis_res, vis, XWALK_COLOR)
    else:
        xwalk_dilated = np.zeros(vis.shape[:2], dtype=bool)

    overlap = False
    h, w = xwalk_dilated.shape
    for (x1, y1, x2, y2) in car_boxes.astype(int):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if np.any(xwalk_dilated[y1:y2, x1:x2]):
            overlap = True
            break

    status_text = f"Car in crosswalk: {overlap}"
    text_color  = OVERLAP_COLOR if overlap else NO_OVERLAP_COLOR
    cv2.putText(vis, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    cv2.imshow("Detection", vis)
    print(f"{fname}: {status_text}")

    if cv2.waitKey(0) & 0xFF == 27:
        break

cv2.destroyAllWindows()
