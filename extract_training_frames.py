import cv2
import os

os.makedirs("training_frames", exist_ok=True)

cap = cv2.VideoCapture("data/3.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# spread ~150 frames evenly across the whole video
NUM_FRAMES = 150
step = total_frames // NUM_FRAMES

saved = 0
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % step == 0:
        cv2.imwrite(f"training_frames/frame_{frame_idx:05d}.jpg", frame)
        saved += 1
    frame_idx += 1

cap.release()
print(f"Saved {saved} frames to training_frames/")