import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import json
from collections import defaultdict

H_direct = np.load("video_to_blueprint.npy")

def video_to_blueprint(px, py):
    p = np.array([px, py, 1.0])
    bp = H_direct @ p
    return bp[0]/bp[2], bp[1]/bp[2]

model = YOLO("yolov8m.pt")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
labels = ["a photo of a man", "a photo of a woman"]

blueprint = cv2.imread("blueprint.png")
bp_h, bp_w = blueprint.shape[:2]

cap = cv2.VideoCapture("data/3.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

all_positions = []               # {frame, track_id, x, y}
gender_votes = defaultdict(list) # track_id -> list of (label, confidence), CONFIDENT ones only

CONF_THRESHOLD = 0.65  # ignore near-coinflip guesses when voting

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
            track_id = int(track_id)
            foot_x, foot_y = (x1 + x2) / 2, y2
            bx, by = video_to_blueprint(foot_x, foot_y)

            if 0 <= bx < bp_w and 0 <= by < bp_h:
                all_positions.append({"frame": frame_count, "track_id": track_id, "x": float(bx), "y": float(by)})

            # classify gender on this frame's crop
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            if crop.size > 0:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(crop_rgb)
                inputs = clip_processor(text=labels, images=pil_img, return_tensors="pt", padding=True)
                outputs = clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)[0]
                pred_idx = probs.argmax().item()
                confidence = probs[pred_idx].item()

                if confidence >= CONF_THRESHOLD:
                    gender_votes[track_id].append((labels[pred_idx], confidence))

    frame_count += 1
    if frame_count % 500 == 0:
        print(f"  {frame_count}/{total_frames} frames done, {len(all_positions)} points so far")

cap.release()

# resolve final gender per track: majority vote among CONFIDENT guesses only
track_gender = {}
for track_id, votes in gender_votes.items():
    man_votes = sum(1 for label, _ in votes if "man" in label)
    woman_votes = sum(1 for label, _ in votes if "woman" in label)
    track_gender[track_id] = "man" if man_votes >= woman_votes else "woman"

# tracks with ZERO confident votes (all frames were uncertain) - default to "unknown"
for p in all_positions:
    tid = p["track_id"]
    p["gender"] = track_gender.get(tid, "unknown")

with open("positions.json", "w") as f:
    json.dump(all_positions, f)

with open("track_gender.json", "w") as f:
    json.dump(track_gender, f)

print(f"\nDone. {frame_count} frames, {len(all_positions)} points, {len(track_gender)} tracks classified.")
unknown_count = sum(1 for tid in set(p['track_id'] for p in all_positions) if tid not in track_gender)
print(f"Tracks with no confident votes (unknown): {unknown_count}")