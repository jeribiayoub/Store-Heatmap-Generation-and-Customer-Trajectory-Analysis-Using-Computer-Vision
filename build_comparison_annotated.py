import cv2
import numpy as np
from ultralytics import YOLO

H_direct = np.load("video_to_blueprint.npy")

def video_to_blueprint(px, py):
    p = np.array([px, py, 1.0])
    bp = H_direct @ p
    return bp[0]/bp[2], bp[1]/bp[2]

model = YOLO("yolov8m.pt")
blueprint = cv2.imread("blueprint.png")
bp_h, bp_w = blueprint.shape[:2]

cap = cv2.VideoCapture("data/3.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

scale = vid_h / bp_h
bp_resized_w = int(bp_w * scale)

out_w = vid_w + bp_resized_w
out = cv2.VideoWriter("comparison_annotated.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, vid_h))

def color_for_id(track_id):
    return (0, 0, 255)  # BGR format - red
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, classes=[0], tracker="botsort.yaml", conf=0.15, verbose=False)
    annotated_frame = results[0].plot()  # original frame WITH boxes + IDs drawn

    canvas_bp = cv2.resize(blueprint, (bp_resized_w, vid_h)).copy()

    boxes = results[0].boxes
    if boxes.id is not None:
        for box, track_id in zip(boxes.xyxy, boxes.id):
            x1, y1, x2, y2 = box.tolist()
            foot_x, foot_y = (x1 + x2) / 2, y2
            
            cv2.circle(annotated_frame, (int(foot_x), int(foot_y)), 6, (0, 0, 255), -1)    
        
            bx, by = video_to_blueprint(foot_x, foot_y)

            if 0 <= bx < bp_w and 0 <= by < bp_h:
                color = color_for_id(int(track_id))
                px, py = int(bx * scale), int(by * scale)
                cv2.circle(canvas_bp, (px, py), 8, color, -1)
                cv2.putText(canvas_bp, str(int(track_id)), (px + 10, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    combined = np.hstack([annotated_frame, canvas_bp])
    out.write(combined)

    frame_idx += 1
    if frame_idx % 500 == 0:
        print(f"  {frame_idx}/{total_frames} frames done")

cap.release()
out.release()
print("Done. Wrote", frame_idx, "frames to comparison_annotated.mp4")