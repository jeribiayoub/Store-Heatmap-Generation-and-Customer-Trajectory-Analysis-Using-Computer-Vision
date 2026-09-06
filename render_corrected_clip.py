import cv2
import numpy as np
import json
from collections import defaultdict

with open("raw_boxes.json") as f:
    raw_boxes = json.load(f)
with open("track_to_chain.json") as f:
    track_to_chain = json.load(f)
with open("chain_gender.json") as f:
    chain_gender = json.load(f)
with open("confirmed_swaps.json") as f:
    confirmed_swaps = json.load(f)

H_direct = np.load("video_to_blueprint.npy")

def video_to_blueprint(px, py):
    p = np.array([px, py, 1.0])
    bp = H_direct @ p
    return bp[0]/bp[2], bp[1]/bp[2]

GENDER_COLOR = {"man": (255, 0, 0), "woman": (203, 92, 255), "unknown": (128, 128, 128)}

def get_effective_chain(chain_id, frame):
    for s in confirmed_swaps:
        if frame > s["frame"]:
            if chain_id == s["chain_a"]:
                return s["chain_b"]
            if chain_id == s["chain_b"]:
                return s["chain_a"]
    return chain_id

boxes_by_frame = defaultdict(list)
for b in raw_boxes:
    boxes_by_frame[b["frame"]].append(b)

blueprint = cv2.imread("blueprint.png")
bp_h, bp_w = blueprint.shape[:2]

START, LENGTH = 2450, 250  # covers before, during, and well after the frame-2548 crossing

cap = cv2.VideoCapture("data/3.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
vid_w, vid_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(cv2.CAP_PROP_POS_FRAMES, START)

scale = vid_h / bp_h
bp_resized_w = int(bp_w * scale)
out = cv2.VideoWriter("swap_corrected_clip.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_w + bp_resized_w, vid_h))

for i in range(LENGTH):
    frame_idx = START + i
    ret, frame = cap.read()
    if not ret:
        break
    canvas_bp = cv2.resize(blueprint, (bp_resized_w, vid_h)).copy()

    for b in boxes_by_frame.get(frame_idx, []):
        raw_chain = track_to_chain.get(str(b["track_id"]))
        if raw_chain is None:
            continue
        effective_chain = get_effective_chain(raw_chain, frame_idx)
        gender = chain_gender.get(str(effective_chain), "unknown")
        color = GENDER_COLOR[gender]

        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{gender} c{effective_chain}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        foot_x, foot_y = (x1 + x2) / 2, y2
        cv2.circle(frame, (int(foot_x), int(foot_y)), 6, color, -1)
        bx, by = video_to_blueprint(foot_x, foot_y)
        if 0 <= bx < bp_w and 0 <= by < bp_h:
            px, py = int(bx * scale), int(by * scale)
            cv2.circle(canvas_bp, (px, py), 8, color, -1)

    out.write(np.hstack([frame, canvas_bp]))

cap.release()
out.release()
print("Done. Saved swap_corrected_clip.mp4")