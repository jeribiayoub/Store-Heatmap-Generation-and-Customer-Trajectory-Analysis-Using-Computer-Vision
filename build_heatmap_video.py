import cv2
import numpy as np
import json
from collections import defaultdict

with open("positions.json") as f:
    positions = json.load(f)

blueprint = cv2.imread("blueprint.png")
bp_h, bp_w = blueprint.shape[:2]

# group positions by frame number for fast lookup
by_frame = defaultdict(list)
max_frame = 0
for p in positions:
    by_frame[p["frame"]].append((int(p["x"]), int(p["y"])))
    max_frame = max(max_frame, p["frame"])

# --- Pass 1: compute the FINAL heatmap to get a fixed color scale ---
final_raw = np.zeros((bp_h, bp_w), dtype=np.float32)
for pts in by_frame.values():
    for x, y in pts:
        if 0 <= x < bp_w and 0 <= y < bp_h:
            final_raw[y, x] += 1
final_smooth = cv2.GaussianBlur(final_raw, (0, 0), sigmaX=20, sigmaY=20)
vmax = final_smooth.max()
print("Fixed color scale max:", vmax)

# --- Pass 2: build the video, growing the heatmap over time ---
STEP = 15        # render a snapshot every 15 raw frames (~0.5s of source video)
OUTPUT_FPS = 20  # playback speed of the output video

out = cv2.VideoWriter(
    "heatmap_growth.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    OUTPUT_FPS,
    (bp_w, bp_h)
)

running = np.zeros((bp_h, bp_w), dtype=np.float32)
snapshots_written = 0

for frame_idx in range(max_frame + 1):
    for x, y in by_frame.get(frame_idx, []):
        if 0 <= x < bp_w and 0 <= y < bp_h:
            running[y, x] += 1

    if frame_idx % STEP == 0:
        smooth = cv2.GaussianBlur(running, (0, 0), sigmaX=20, sigmaY=20)
        norm = np.clip(smooth / vmax * 255, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        mask = norm > 10
        canvas = blueprint.copy()
        canvas[mask] = cv2.addWeighted(blueprint, 0.4, color, 0.6, 0)[mask]

        out.write(canvas)
        snapshots_written += 1

out.release()
print(f"Done. Wrote {snapshots_written} snapshots to heatmap_growth.mp4")