import cv2
import numpy as np
import torch
from ultralytics import YOLO
from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.utils import WEIGHTS

model_yolo = YOLO("runs/detect/store_finetune/weights/best.pt")
reid_model = ReidAutoBackend(weights=WEIGHTS / "osnet_x0_25_msmt17.pt", device=torch.device("cuda:0")).model

cap = cv2.VideoCapture("data/3.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
ret, frame = cap.read()
cap.release()

results = model_yolo(frame, classes=[0], device=0, verbose=False)
boxes = results[0].boxes.xyxy.cpu().numpy()

print("Detections found:", len(boxes))

if len(boxes) > 0:
    embeddings = reid_model.get_features(boxes, frame)
    print("Embeddings shape:", embeddings.shape)
    print("First embedding (first 5 values):", embeddings[0][:5])

    # sanity check: compare person 0 against person 1 (should be LESS similar than
    # comparing person 0 against itself, obviously - just confirming the math works)
    if len(embeddings) > 1:
        sim = np.dot(embeddings[0], embeddings[1])
        print(f"\nCosine similarity between person 0 and person 1: {sim:.3f}")