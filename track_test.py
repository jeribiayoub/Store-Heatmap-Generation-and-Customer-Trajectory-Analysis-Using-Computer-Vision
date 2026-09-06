import cv2
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture("data/3.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# start partway into the video where we know people are visible
cap.set(cv2.CAP_PROP_POS_FRAMES, 500)

out = cv2.VideoWriter("track_test_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

frame_count = 0
max_frames = 300  # about 10 seconds of video at 30fps

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, classes=[0], tracker="botsort.yaml", verbose=False)
    annotated = results[0].plot()  # now boxes will include ID numbers
    out.write(annotated)

    frame_count += 1

cap.release()
out.release()
print("Done. Processed", frame_count, "frames.")