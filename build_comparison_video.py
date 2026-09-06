import cv2
import numpy as np
import json
from collections import defaultdict

with open("positions_stitched_smoothed.json") as f:
    positions = json.load(f)

blueprint = cv2.imread("blueprint.png")
bp_h, bp_w = blueprint.shape[:2]

# group by frame, keep track_id so we can color/label each person consistently
by_frame = defaultdict(list)
for p in positions:
    by_frame[p["frame"]].append((p["track_id"], p["x"], p["y"]))

cap = cv2.VideoCapture("data/3.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# resize blueprint to match the video's height so they sit side by side cleanly
scale = vid_h / bp_h
bp_resized_w = int(bp_w * scale)
blueprint_resized = cv2.resize(blueprint, (bp_resized_w, vid_h))

out_w = vid_w + bp_resized_w
out = cv2.VideoWriter("comparison_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, vid_h))

def color_for_id(track_id):
    return (0, 0, 255)  # BGR format - red

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    canvas_bp = blueprint_resized.copy()
    for track_id, x, y in by_frame.get(frame_idx, []):
        color = color_for_id(track_id)
        px, py = int(x * scale), int(y * scale)
        cv2.circle(canvas_bp, (px, py), 8, color, -1)
        cv2.putText(canvas_bp, str(track_id), (px + 10, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    combined = np.hstack([frame, canvas_bp])
    out.write(combined)
    frame_idx += 1

cap.release()
out.release()
print("Done. Wrote", frame_idx, "frames to comparison_video.mp4")