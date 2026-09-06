import cv2
import numpy as np
from ultralytics import YOLO
from boxmot import OcSort
from pathlib import Path

model = YOLO("runs/detect/store_finetune/weights/best.pt")
tracker = OcSort()

cap = cv2.VideoCapture("data/3.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
ret, frame = cap.read()
cap.release()

results = model(frame, classes=[0], verbose=False)
boxes = results[0].boxes.xyxy.cpu().numpy()
confs = results[0].boxes.conf.cpu().numpy()
clss = results[0].boxes.cls.cpu().numpy()

print("Detections found:", len(boxes))

if len(boxes) > 0:
    dets = np.concatenate([boxes, confs[:, None], clss[:, None]], axis=1)
else:
    dets = np.empty((0, 6))

tracks = tracker.update(dets, frame)

print("\nTracker output shape:", tracks.shape)
print("First row (if any):", tracks[0] if len(tracks) > 0 else "no tracks")