import cv2
import numpy as np

cap = cv2.VideoCapture("data/3.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

SAMPLE_EVERY = 15  # grab a frame every 15 frames, spread across the whole video
samples = []

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % SAMPLE_EVERY == 0:
        samples.append(frame)
    frame_idx += 1

cap.release()
print(f"Collected {len(samples)} sample frames for background estimation")

stack = np.stack(samples, axis=0)  # shape: (num_samples, H, W, 3)
median_frame = np.median(stack, axis=0).astype(np.uint8)

cv2.imwrite("clean_background.jpg", median_frame)
print("Saved clean_background.jpg")