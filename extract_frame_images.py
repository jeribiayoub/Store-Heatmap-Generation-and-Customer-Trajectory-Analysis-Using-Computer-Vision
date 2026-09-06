import cv2
import os
import json
from collections import defaultdict

with open("raw_boxes.json") as f:
    raw_boxes = json.load(f)
with open("track_to_chain.json") as f:
    track_to_chain = json.load(f)

boxes_by_frame = defaultdict(list)
for b in raw_boxes:
    boxes_by_frame[b["frame"]].append(b)

os.makedirs("frames_4300_4400", exist_ok=True)
START, END = 4300, 4400

cap = cv2.VideoCapture("data/3.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, START)

def color_for_id(chain_id):
    import numpy as np
    np.random.seed(chain_id)
    return tuple(int(c) for c in np.random.randint(50, 255, size=3))

for frame_idx in range(START, END):
    ret, frame = cap.read()
    if not ret:
        break

    for b in boxes_by_frame.get(frame_idx, []):
        chain_id = track_to_chain.get(str(b["track_id"]))
        if chain_id is None:
            continue
        color = color_for_id(chain_id)
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{chain_id}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, f"FRAME {frame_idx}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(f"frames_4300_4400/f{frame_idx}.jpg", frame)

cap.release()
print(f"Saved frames {START}-{END} to frames_4300_4400/")