# view_predictions.py
import cv2
import glob
import os
from ultralytics import YOLO
import numpy as np
from pathlib import Path

# ---------- config ----------
IMG_DIR     = Path("test-data")          # dir that holds the .png images
CAR_WEIGHTS = Path("cars.pt")
SIG_WEIGHTS = Path("signals.pt")
CROSS_WEIGHTS = Path("cross.pt")
IMG_SIZE    = 640                        # inference resolution (kept square)
FONT        = cv2.FONT_HERSHEY_SIMPLEX
# -----------------------------

# Load models once
car_model   = YOLO(str(CAR_WEIGHTS))
sig_model   = YOLO(str(SIG_WEIGHTS))
cross_model = YOLO(str(CROSS_WEIGHTS))

# Convenience: colour palette (BGR)
COLORS = {
    "car":      (36, 255, 12),   # lime
    "signal":   (0, 255, 255),   # yellow
    "cross":    (255, 0, 0)      # blue mask overlay
}

def draw_boxes(result, frame, color_name):
    """Draw bounding boxes + labels from a YOLO result on frame."""
    names = result.names
    for box in result.boxes:
        cls_id = int(box.cls)
        label  = names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[color_name], 2)
        txt = f"{label} {box.conf.item():.2f}"
        cv2.putText(frame, txt, (x1, y1 - 5), FONT, 0.5, COLORS[color_name], 1, cv2.LINE_AA)

def draw_masks(result, frame):
    """Overlay segmentation masks (binary) with transparency."""
    if not result.masks:
        return
    masks = result.masks.data.cpu().numpy()  # (num, H, W)
    for m in masks:
        colored = np.zeros_like(frame, dtype=np.uint8)
        colored[m > 0.5] = COLORS["cross"]   # solid colour where mask == 1
        frame[:] = cv2.addWeighted(frame, 1.0, colored, 0.4, 0)  # 40 % overlay

    # also draw box / label for each instance
    draw_boxes(result, frame, "cross")

def process_image(img_path):
    """Run three models and return one annotated 640×640 frame."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read {img_path}")

    # --- NEW: force 640×640 so masks & boxes match ---
    img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_LINEAR)
    # -------------------------------------------------

    # Run all three models
    car_res   = car_model.predict(img_bgr, imgsz=IMG_SIZE, verbose=False)[0]
    sig_res   = sig_model.predict(img_bgr, imgsz=IMG_SIZE, verbose=False)[0]
    cross_res = cross_model.predict(img_bgr, imgsz=IMG_SIZE, verbose=False)[0]

    # Draw everything on a copy
    out = img_bgr.copy()
    draw_boxes(car_res,   out, "car")
    draw_boxes(sig_res,   out, "signal")
    draw_masks(cross_res, out)

    return out


def main():
    img_files = sorted(glob.glob(str(IMG_DIR / "*.png")))
    if not img_files:
        print(f"No .png images found in {IMG_DIR}")
        return

    idx = 0
    while idx < len(img_files):
        annotated = process_image(img_files[idx])
        cv2.imshow("YOLO predictions (p = next, q = quit)", annotated)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("p"):
            idx += 1
        elif key == ord("q"):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
