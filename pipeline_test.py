import cv2
import numpy as np
from ultralytics import YOLO

# load both saved transforms
H_direct = np.load("video_to_blueprint.npy")

def video_to_blueprint(px, py):
    p = np.array([px, py, 1.0])
    bp = H_direct @ p
    return bp[0]/bp[2], bp[1]/bp[2]


model = YOLO("yolov8m.pt")
blueprint = cv2.imread("blueprint.png")

cap = cv2.VideoCapture("data/3.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 500)

frame_count = 0
max_frames = 150  # 5 seconds, just to test

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, classes=[0], tracker="botsort.yaml", conf=0.15, verbose=False)
    boxes = results[0].boxes

    canvas = blueprint.copy()
    if boxes.id is not None:
        for box, track_id in zip(boxes.xyxy, boxes.id):
            x1, y1, x2, y2 = box.tolist()
            foot_x, foot_y = (x1 + x2) / 2, y2
            bx, by = video_to_blueprint(foot_x, foot_y)
            if 0 <= bx < canvas.shape[1] and 0 <= by < canvas.shape[0]:
                cv2.circle(canvas, (int(bx), int(by)), 8, (0, 0, 255), -1)
                cv2.putText(canvas, str(int(track_id)), (int(bx)+10, int(by)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    cv2.imwrite(f"pipeline_frame_{frame_count:03d}.jpg", canvas)
    frame_count += 1

cap.release()
print("Done, wrote", frame_count, "blueprint frames.")