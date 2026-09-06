import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO

model_yolo = YOLO("yolov8m.pt")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

labels = ["a photo of a man", "a photo of a woman"]

# spread across the whole video so we sample different people, not the same few repeatedly
test_frames = [200, 800, 1400, 2000, 2600, 3200, 3800, 4400, 5000, 5600, 6200]

cap = cv2.VideoCapture("data/3.mp4")

person_count = 0
for frame_num in test_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        continue

    results = model_yolo(frame, classes=[0], device=0, verbose=False)

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        filename = f"sample_f{frame_num}_p{person_count}.jpg"
        cv2.imwrite(filename, crop)

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        inputs = clip_processor(text=labels, images=pil_img, return_tensors="pt", padding=True).to("cuda")
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        pred_idx = probs.argmax().item()

        print(f"{filename}: {labels[pred_idx]} ({probs[pred_idx]*100:.1f}%)")
        person_count += 1

cap.release()
print(f"\nTotal people sampled: {person_count}")