import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import json
from collections import defaultdict

with open("raw_boxes.json") as f:
    raw_boxes = json.load(f)
with open("track_to_chain.json") as f:
    track_to_chain = json.load(f)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

boxes_by_frame = defaultdict(list)
for b in raw_boxes:
    boxes_by_frame[b["frame"]].append(b)

# for each chain, keep embeddings from a SAMPLE of frames (not every single one - fast + enough signal)
chain_embeddings = defaultdict(list)
SAMPLE_EVERY = 5  # only embed every 5th frame of each track, to keep this fast

cap = cv2.VideoCapture("data/3.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % SAMPLE_EVERY == 0:
        for b in boxes_by_frame.get(frame_idx, []):
            chain_id = track_to_chain.get(str(b["track_id"]))
            if chain_id is None:
                continue
            x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            inputs = clip_processor(images=pil_img, return_tensors="pt").to("cuda")
            with_grad_off = clip_model.get_image_features(**inputs)
            if hasattr(with_grad_off, "image_embeds"):
                embedding_tensor = with_grad_off.image_embeds
            elif hasattr(with_grad_off, "pooler_output"):
                embedding_tensor = with_grad_off.pooler_output
            else:
                embedding_tensor = with_grad_off
            embedding = embedding_tensor.detach().cpu().numpy()[0]
            chain_embeddings[chain_id].append(embedding)

    frame_idx += 1
    if frame_idx % 1000 == 0:
        print(f"  {frame_idx}/{total_frames} frames done")

cap.release()

# average all sampled embeddings into one signature per chain
chain_signatures = {}
for chain_id, embs in chain_embeddings.items():
    avg = np.mean(embs, axis=0)
    avg = avg / np.linalg.norm(avg)  # normalize, so later comparisons are pure cosine similarity
    chain_signatures[chain_id] = avg.tolist()

with open("chain_signatures.json", "w") as f:
    json.dump(chain_signatures, f)

print(f"\nDone. Built appearance signatures for {len(chain_signatures)} chains.")