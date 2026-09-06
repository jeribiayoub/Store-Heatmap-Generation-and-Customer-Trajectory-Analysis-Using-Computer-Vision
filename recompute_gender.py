import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import json
from collections import defaultdict

with open("raw_boxes.json") as f:
    raw_boxes = json.load(f)

boxes_by_frame = defaultdict(list)
for b in raw_boxes:
    boxes_by_frame[b["frame"]].append(b)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
labels = ["a photo of a man", "a photo of a woman"]

cap = cv2.VideoCapture("data/3.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

conf_sum = defaultdict(lambda: {"man": 0.0, "woman": 0.0})

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    for b in boxes_by_frame.get(frame_idx, []):
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        inputs = clip_processor(text=labels, images=pil_img, return_tensors="pt", padding=True).to("cuda")
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

        tid = b["track_id"]
        conf_sum[tid]["man"] += probs[0].item()
        conf_sum[tid]["woman"] += probs[1].item()

    frame_idx += 1
    if frame_idx % 1000 == 0:
        print(f"  {frame_idx}/{total_frames} frames done")

cap.release()

MARGIN = 0.1

track_gender = {}
for tid, sums in conf_sum.items():
    total = sums["man"] + sums["woman"]
    if total == 0:
        continue
    man_share = sums["man"] / total
    woman_share = sums["woman"] / total
    if man_share - woman_share > MARGIN:
        track_gender[tid] = "man"
    elif woman_share - man_share > MARGIN:
        track_gender[tid] = "woman"
    else:
        track_gender[tid] = "unknown"

with open("track_conf_sums.json", "w") as f:
    json.dump(conf_sum, f)

with open("track_gender.json", "w") as f:
    json.dump(track_gender, f)

man_n = sum(1 for v in track_gender.values() if v == "man")
woman_n = sum(1 for v in track_gender.values() if v == "woman")
unk_n = sum(1 for v in track_gender.values() if v == "unknown")
print(f"Done. {len(track_gender)} tracks re-classified.")
print(f"man={man_n}, woman={woman_n}, unknown={unk_n}")