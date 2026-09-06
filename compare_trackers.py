import cv2
import numpy as np
from ultralytics import YOLO
from boxmot import OcSort

model = YOLO("runs/detect/store_finetune/weights/best.pt")

# test clips: (start_frame, num_frames) around your known crossing events
clips = [
    (600, 200),   # covers ~22s crossing
    (2450, 200),  # covers ~85s confirmed swap
]

def run_botsort_reid(start, length, out_path):
    cap = cv2.VideoCapture("data/3.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for _ in range(length):
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, persist=True, classes=[0], tracker="mytracker.yaml", verbose=False)
        annotated = results[0].plot()
        out.write(annotated)
    cap.release()
    out.release()

def run_ocsort(start, length, out_path):
    cap = cv2.VideoCapture("data/3.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracker = OcSort()
    for _ in range(length):
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, classes=[0], verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy()
        dets = np.concatenate([boxes, confs[:, None], clss[:, None]], axis=1) if len(boxes) > 0 else np.empty((0, 6))

        tracks = tracker.update(dets, frame)
        for t in tracks:
            x1, y1, x2, y2, tid = t[0], t[1], t[2], t[3], int(t[4])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {tid}", (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        out.write(frame)
    cap.release()
    out.release()

for start, length in clips:
    print(f"Processing clip starting at frame {start}...")
    run_botsort_reid(start, length, f"clip_{start}_botsort.mp4")
    run_ocsort(start, length, f"clip_{start}_ocsort.mp4")

print("Done. Compare clip_600_botsort.mp4 vs clip_600_ocsort.mp4, and clip_2450_botsort.mp4 vs clip_2450_ocsort.mp4")