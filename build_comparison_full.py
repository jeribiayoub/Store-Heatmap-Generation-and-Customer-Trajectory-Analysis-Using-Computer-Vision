import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

H_direct = np.load("video_to_blueprint.npy")

def video_to_blueprint(px, py):
    p = np.array([px, py, 1.0])
    bp = H_direct @ p
    return bp[0]/bp[2], bp[1]/bp[2]

GENDER_COLOR = {
    "a photo of a man": (255, 0, 0),      # BGR - blue
    "a photo of a woman": (203, 92, 255), # BGR - pink
}
labels = ["a photo of a man", "a photo of a woman"]

model = YOLO("yolov8m.pt")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

blueprint = cv2.imread("blueprint.png")
bp_h, bp_w = blueprint.shape[:2]

cap = cv2.VideoCapture("data/3.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

scale = vid_h / bp_h
bp_resized_w = int(bp_w * scale)
out_w = vid_w + bp_resized_w
out = cv2.VideoWriter("comparison_full.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, vid_h))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, classes=[0], tracker="botsort.yaml", verbose=False)
    boxes = results[0].boxes
    canvas_bp = cv2.resize(blueprint, (bp_resized_w, vid_h)).copy()

    if boxes.id is not None:
        for box, track_id in zip(boxes.xyxy, boxes.id):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            inputs = clip_processor(text=labels, images=pil_img, return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
            pred_label = labels[probs.argmax().item()]
            color = GENDER_COLOR[pred_label]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            foot_x, foot_y = (x1 + x2) / 2, y2
            cv2.circle(frame, (int(foot_x), int(foot_y)), 6, color, -1)

            bx, by = video_to_blueprint(foot_x, foot_y)
            if 0 <= bx < bp_w and 0 <= by < bp_h:
                px, py = int(bx * scale), int(by * scale)
                cv2.circle(canvas_bp, (px, py), 8, color, -1)

    combined = np.hstack([frame, canvas_bp])
    out.write(combined)

    frame_idx += 1
    if frame_idx % 500 == 0:
        print(f"  {frame_idx}/{total_frames} frames done")

cap.release()
out.release()
print("Done. Wrote", frame_idx, "frames to comparison_full.mp4")