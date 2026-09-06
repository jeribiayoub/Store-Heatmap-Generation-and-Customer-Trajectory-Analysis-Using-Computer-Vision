import cv2
import os
import json
from collections import defaultdict

with open("raw_boxes.json") as f:
    raw_boxes = json.load(f)
with open("track_to_chain.json") as f:
    track_to_chain = json.load(f)

os.makedirs("labeling_crops", exist_ok=True)

boxes_by_chain = defaultdict(list)
for b in raw_boxes:
    chain_id = track_to_chain.get(str(b["track_id"]))
    if chain_id is not None:
        boxes_by_chain[chain_id].append(b)

CROPS_PER_CHAIN = 8  # spread across each person's visit, for variety in pose/angle
MIN_CROP_HEIGHT = 80

cap = cv2.VideoCapture("data/3.mp4")

saved = 0
for chain_id, boxes in boxes_by_chain.items():
    boxes.sort(key=lambda b: b["frame"])
    # pick evenly spaced boxes across this chain's lifetime
    step = max(1, len(boxes) // CROPS_PER_CHAIN)
    selected = boxes[::step][:CROPS_PER_CHAIN]

    for b in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, b["frame"])
        ret, frame = cap.read()
        if not ret:
            continue
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        if crop.shape[0] < MIN_CROP_HEIGHT:  # skip if too small/blurry-by-distance
            continue
        filename = f"labeling_crops/chain{chain_id}_frame{b['frame']}.jpg"
        cv2.imwrite(filename, crop)
        saved += 1

cap.release()
print(f"Saved {saved} crops to labeling_crops/ across {len(boxes_by_chain)} chains")