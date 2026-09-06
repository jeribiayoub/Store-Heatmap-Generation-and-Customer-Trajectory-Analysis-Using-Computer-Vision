import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO

model_yolo = YOLO("yolov8m.pt")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

labels = ["a photo of a man", "a photo of a woman"]

cap = cv2.VideoCapture("data/3.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
ret, frame = cap.read()
cap.release()

results = model_yolo(frame, classes=[0], verbose=False)

for i, box in enumerate(results[0].boxes.xyxy):
    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
    crop = frame[y1:y2, x1:x2]
    cv2.imwrite(f"person_crop_{i}.jpg", crop)  # look at these to sanity-check quality

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)

    inputs = clip_processor(text=labels, images=pil_img, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]

    pred_idx = probs.argmax().item()
    print(f"Person {i}: {labels[pred_idx]} ({probs[pred_idx]*100:.1f}% confidence)")