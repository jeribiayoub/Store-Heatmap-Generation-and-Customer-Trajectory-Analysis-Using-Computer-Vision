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

def color_for_id(chain_id):
    np.random.seed(chain_id)
    return tuple(int(c) for c in np.random.randint(50, 255, size=3))

cap_meta = cv2.VideoCapture("data/3.mp4")
fps = cap_meta.get(cv2.CAP_PROP_FPS)
vid_w, vid_h = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_meta.release()

CENTERS = [660, 2250, 2550, 4110]  # 22s, 1:15, 1:25, 2:17
BEFORE, AFTER = 90, 90

for center in CENTERS:
    start = max(0, center - BEFORE)
    length = BEFORE + AFTER

    cap = cv2.VideoCapture("data/3.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = cv2.VideoWriter(f"clip_{center}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_w, vid_h))

    for i in range(length):
        frame_idx = start + i
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
            text_x = max(5, min(x1, vid_w - 40))
            cv2.putText(frame, f"{chain_id}", (text_x, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if frame_idx == center:
            cv2.putText(frame, f"<<< frame {center} >>>", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved clip_{center}.mp4")

print("Done.")