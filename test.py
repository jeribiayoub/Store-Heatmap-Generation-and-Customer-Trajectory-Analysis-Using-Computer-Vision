import cv2
cap = cv2.VideoCapture("data/3.mp4")
print("Opened:", cap.isOpened())
print("Frame count:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Resolution:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()