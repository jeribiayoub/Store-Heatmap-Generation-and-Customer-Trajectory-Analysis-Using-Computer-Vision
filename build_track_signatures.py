import cv2
import numpy as np
import torch
from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.utils import WEIGHTS
import json
from collections import defaultdict

with open("raw_boxes.json") as f:
    raw_boxes = json.load(f)

reid_model = ReidAutoBackend(weights=WEIGHTS / "osnet_x0_25_msmt17.pt", device=torch.device("cuda:0")).model

boxes_by_frame = defaultdict(list)
for b in raw_boxes:
    boxes_by_frame[b["frame"]].append(b)

track_embeddings = defaultdict(list)
SAMPLE_EVERY = 5

cap = cv2.VideoCapture("data/3.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % SAMPLE_EVERY == 0:
        frame_boxes = boxes_by_frame.get(frame_idx, [])
        if frame_boxes:
            xyxys = np.array([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in frame_boxes])
            embeddings = reid_model.get_features(xyxys, frame)
            for b, emb in zip(frame_boxes, embeddings):
                track_embeddings[b["track_id"]].append(emb)

    frame_idx += 1
    if frame_idx % 1000 == 0:
        print(f"  {frame_idx}/{total_frames} frames done")

cap.release()

track_signatures = {}
for tid, embs in track_embeddings.items():
    avg = np.mean(embs, axis=0)
    avg = avg / np.linalg.norm(avg)
    track_signatures[tid] = avg.tolist()

with open("track_signatures_raw.json", "w") as f:
    json.dump(track_signatures, f)

print(f"\nDone. Built signatures for {len(track_signatures)} raw tracks.")