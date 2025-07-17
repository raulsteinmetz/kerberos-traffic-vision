import os
import cv2
import numpy as np
from ultralytics import YOLO

# load trained instance segmentation models (yolo11)
car_model = YOLO('./car-seg-y11/model_finetuned/weights/best.pt')
crosswalk_model = YOLO('./cross-walk-seg-y11/model_finetuned/weights/best.pt')

# input size size
WIDTH, HEIGHT = 640, 640
image_dir = '../traffic-detection/test-data/'

CAR_COLOR = (0, 0, 255)
XWALK_COLOR = (0, 255, 0)
ALPHA = 0.5

def overlay_masks(result, base_img, color):
    if not hasattr(result, 'masks') or result.masks is None:
        return base_img
    mask_data = result.masks.data.cpu().numpy()  # shape: (N, H, W)
    for m in mask_data:
        m_bin = (m > 0.5).astype(np.uint8)
        colored = np.zeros_like(base_img, dtype=np.uint8)
        colored[:] = color
        base_img = np.where(
            m_bin[..., None],
            cv2.addWeighted(base_img, 1 - ALPHA, colored, ALPHA, 0),
            base_img
        )
    return base_img

for filename in sorted(os.listdir(image_dir)):
    path = os.path.join(image_dir, filename)
    img = cv2.imread(path)
    if img is None:
        continue
    img = cv2.resize(img, (WIDTH, HEIGHT))
    result_img = img.copy()

    car_r = car_model.predict(source=img, imgsz=(WIDTH, HEIGHT), conf=0.25)[0]
    cw_r = crosswalk_model.predict(source=img, imgsz=(WIDTH, HEIGHT), conf=0.25)[0]

    result_img = overlay_masks(car_r, result_img, CAR_COLOR)
    result_img = overlay_masks(cw_r, result_img, XWALK_COLOR)

    for r, clr in [(car_r, CAR_COLOR), (cw_r, XWALK_COLOR)]:
        if hasattr(r, 'boxes') and r.boxes is not None:
            for box in r.boxes.xyxy.cpu().numpy().astype(int):
                x1, y1, x2, y2 = box
                cv2.rectangle(result_img, (x1, y1), (x2, y2), clr, 2)

    cv2.imshow('Segmentation Overlay', result_img)

    # press ESC to exit early, or any key to advance
    key = cv2.waitKey(0) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
print("Visual demo complete.")
