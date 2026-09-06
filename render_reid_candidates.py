import cv2
import numpy as np
import json
from collections import defaultdict

with open("raw_boxes.json") as f:
    raw_boxes = json.load(f)
with open("track_to_chain.json") as f:
    track_to_chain = json.load(f)

boxes_by_chain_frame = defaultdict(dict)
for b in raw_boxes:
    chain_id = track_to_chain.get(str(b["track_id"]))
    if chain_id is not None:
        boxes_by_chain_frame[chain_id][b["frame"]] = b

# top 4 candidates to verify
candidates = [
    (3, 25),
    (28, 42),
    (2, 15),
    (1, 42),
]

CLIP_LEN = 90  # ~3 seconds at 30fps

cap_meta = cv2.VideoCapture("data/3.mp4")
fps = cap_meta.get(cv2.CAP_PROP_FPS)
vid_w, vid_h = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_meta.release()

for cid_a, cid_b in candidates:
    frames_a = sorted(boxes_by_chain_frame[cid_a].keys())
    frames_b = sorted(boxes_by_chain_frame[cid_b].keys())

    a_end_frames = frames_a[-CLIP_LEN:]  # last ~3s of chain A
    b_start_frames = frames_b[:CLIP_LEN]  # first ~3s of chain B

    out_path = f"reid_candidate_{cid_a}_{cid_b}.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_w, vid_h))

    cap = cv2.VideoCapture("data/3.mp4")

    def write_segment(frame_list, chain_id, label):
        for f in frame_list:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            if not ret:
                continue
            b = boxes_by_chain_frame[chain_id][f]
            x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            out.write(frame)

    write_segment(a_end_frames, cid_a, f"CHAIN {cid_a} - ending")
    write_segment(b_start_frames, cid_b, f"CHAIN {cid_b} - starting")

    cap.release()
    out.release()
    print(f"Saved {out_path}")

print("Done.")