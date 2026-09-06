import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # will auto-download the first time you run this (~6MB)

cap = cv2.VideoCapture("data/3.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 500)  # jump to frame 500
ret, frame = cap.read()
cap.release()

if not ret:
    print("Couldn't read frame")
else:
    results = model(frame, classes=[0])  # class 0 = person
    annotated = results[0].plot()        # draws boxes on the frame
    cv2.imwrite("detect_test_output.jpg", annotated)
    print("Detections found:", len(results[0].boxes))