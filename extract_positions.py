import cv2
import numpy as np
from ultralytics import YOLO
import json

H_direct = np.load("video_to_blueprint.npy")

def video_to_blueprint(px, py):
    p = np.array([px, py, 1.0])
    bp = H_direct @ p
    return bp[0]/bp[2], bp[1]/bp[2]

model = YOLO("yolov8m.pt")
blueprint = cv2.imread("blueprint.png")
bp_h, bp_w = blueprint.shape[:2]

cap = cv2.VideoCapture("data/3.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

all_positions = []  # list of dicts: {frame, track_id, x, y}
frame_count = 0

print(f"Processing {total_frames} frames...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, classes=[0], tracker="botsort.yaml", verbose=False)
    boxes = results[0].boxes

    if boxes.id is not None:
        for box, track_id in zip(boxes.xyxy, boxes.id):
            x1, y1, x2, y2 = box.tolist()
            foot_x, foot_y = (x1 + x2) / 2, y2
            bx, by = video_to_blueprint(foot_x, foot_y)

            # only keep points that land within the blueprint canvas
            if 0 <= bx < bp_w and 0 <= by < bp_h:
                all_positions.append({
                    "frame": frame_count,
                    "track_id": int(track_id),
                    "x": float(bx),
                    "y": float(by),
                })

    frame_count += 1
    if frame_count % 500 == 0:
        print(f"  {frame_count}/{total_frames} frames done, {len(all_positions)} points so far")

cap.release()

with open("positions.json", "w") as f:
    json.dump(all_positions, f)

print(f"\nDone. {frame_count} frames processed, {len(all_positions)} total position points saved to positions.json")