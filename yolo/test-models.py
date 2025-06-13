import cv2
import os
import time
from ultralytics import YOLO


model_path = './runs_y11/cars/weights/best.pt'
model = YOLO(model_path)

image_dir = '../traffic-detection/test-data/'

image_files = sorted([
    os.path.join(image_dir, f) for f in os.listdir(image_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

for image_path in image_files:
    results = model(image_path)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv11 Inference', annotated_frame)
    key = cv2.waitKey(200)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
