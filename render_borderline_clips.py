import cv2
import numpy as np
import json
from collections import defaultdict

with open("raw_boxes.json") as f:
    raw_boxes = json.load(f)
with open("track_to_chain.json") as f:
    track_to_chain = json.load(f)

boxes_by_frame = defaultdict(list)
for b in raw_boxes:
    boxes_by_frame[b["frame"]].append(b)

# the 4 borderline cases: (chain_a, chain_b, crossing_frame)
cases = [
    (15, 17, 2265),
    (15, 18, 2560),
    (1, 28, 3112),
    (28, 37, 3848),
]

BEFORE, AFTER = 100, 150  # frames of context around each crossing

cap_meta = cv2.VideoCapture("data/3.mp4")
fps = cap_meta.get(cv2.CAP_PROP_FPS)
vid_w, vid_h = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_meta.release()

for cid_a, cid_b, crossing_frame in cases:
    start = max(0, crossing_frame - BEFORE)
    length = BEFORE + AFTER

    cap = cv2.VideoCapture("data/3.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out_path = f"borderline_{cid_a}_{cid_b}_{crossing_frame}.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_w, vid_h))

    for i in range(length):
        frame_idx = start + i
        ret, frame = cap.read()
        if not ret:
            break

        for b in boxes_by_frame.get(frame_idx, []):
            chain_id = track_to_chain.get(str(b["track_id"]))
            x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])

            if chain_id == cid_a:
                color = (0, 0, 255)   # red
                label = f"CHAIN A ({cid_a})"
            elif chain_id == cid_b:
                color = (255, 0, 0)   # blue
                label = f"CHAIN B ({cid_b})"
            else:
                color = (150, 150, 150)  # gray for everyone else in frame
                label = ""

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if label:
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # mark the exact crossing frame clearly
        if frame_idx == crossing_frame:
            cv2.putText(frame, "<<< CROSSING FRAME >>>", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved {out_path}")

print("Done.")