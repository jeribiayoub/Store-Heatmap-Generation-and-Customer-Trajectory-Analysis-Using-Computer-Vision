import cv2

from ultralytics import YOLO

models_to_test = {
    "yolov8m_original": YOLO("yolov8m.pt"),
    "yolov8m_finetuned": YOLO("runs/detect/store_finetune/weights/best.pt"),
}

cap = cv2.VideoCapture("data/3.mp4")

# pick a handful of frames where you KNOW a client was missed before
test_frames = [593,592,570, 569,561, 562, 557, 553, 554, 545, 539, 538, 533, 532, 531]  # <-- adjust to frames where you noticed misses

for frame_num in test_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        continue

    for name, model in models_to_test.items():
        results = model(frame, classes=[0], verbose=False)  # default conf, no tracking needed for this test
        annotated = results[0].plot()
        count = len(results[0].boxes)
        filename = f"compare_frame{frame_num}_{name}.jpg"
        cv2.imwrite(filename, annotated)
        print(f"Frame {frame_num} | {name}: {count} detections -> {filename}")

cap.release()