import cv2
import numpy as np
import torch
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
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

labels = ["a photo of a man", "a photo of a woman"]

def classify_gender_zeroshot(crop):
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    inputs = clip_processor(text=labels, images=pil_img, return_tensors="pt", padding=True).to("cuda")
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]
    return probs[0].item(), probs[1].item()  # (P(man), P(woman))

blueprint = cv2.imread("blueprint.png")
bp_h, bp_w = blueprint.shape[:2]

cap = cv2.VideoCapture("data/3.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

all_positions = []
all_boxes = []
gender_conf_sum = defaultdict(lambda: {"man": 0.0, "woman": 0.0})

frame_count = 0
print(f"Processing {total_frames} frames...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, classes=[0], tracker="mytracker.yaml", device=0, verbose=False)
    boxes = results[0].boxes

    if boxes.id is not None:
        for box, track_id in zip(boxes.xyxy, boxes.id):
            x1, y1, x2, y2 = box.tolist()
            track_id = int(track_id)
            foot_x, foot_y = (x1 + x2) / 2, y2
            bx, by = video_to_blueprint(foot_x, foot_y)

            if 0 <= bx < bp_w and 0 <= by < bp_h:
                all_positions.append({"frame": frame_count, "track_id": track_id, "x": float(bx), "y": float(by)})

            all_boxes.append({"frame": frame_count, "track_id": track_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2})

            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            if crop.size > 0:
                man_prob, woman_prob = classify_gender_zeroshot(crop)
                gender_conf_sum[track_id]["man"] += man_prob
                gender_conf_sum[track_id]["woman"] += woman_prob

    frame_count += 1
    if frame_count % 500 == 0:
        print(f"  {frame_count}/{total_frames} frames done")

cap.release()

MARGIN = 0.1
track_gender = {}
for track_id, sums in gender_conf_sum.items():
    total = sums["man"] + sums["woman"]
    if total == 0:
        continue
    man_share = sums["man"] / total
    woman_share = sums["woman"] / total
    if man_share - woman_share > MARGIN:
        track_gender[track_id] = "man"
    elif woman_share - man_share > MARGIN:
        track_gender[track_id] = "woman"
    else:
        track_gender[track_id] = "unknown"

for p in all_positions:
    p["gender"] = track_gender.get(p["track_id"], "unknown")

with open("positions.json", "w") as f:
    json.dump(all_positions, f)
with open("raw_boxes.json", "w") as f:
    json.dump(all_boxes, f)
with open("track_gender.json", "w") as f:
    json.dump(track_gender, f)
with open("track_conf_sums.json", "w") as f:
    json.dump(gender_conf_sum, f)

man_n = sum(1 for v in track_gender.values() if v == "man")
woman_n = sum(1 for v in track_gender.values() if v == "woman")
unk_n = sum(1 for v in track_gender.values() if v == "unknown")
print(f"\nDone. {frame_count} frames, {len(all_positions)} points, {len(all_boxes)} boxes.")
print(f"man={man_n}, woman={woman_n}, unknown={unk_n}")