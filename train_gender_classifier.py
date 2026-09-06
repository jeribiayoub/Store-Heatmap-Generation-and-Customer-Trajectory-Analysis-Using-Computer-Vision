import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
from sklearn.model_selection import cross_val_score

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_embedding(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    inputs = clip_processor(images=pil_img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = clip_model.get_image_features(**inputs)
    if hasattr(output, "image_embeds"):
        emb = output.image_embeds
    elif hasattr(output, "pooler_output"):
        emb = output.pooler_output
    else:
        emb = output
    return emb.detach().cpu().numpy()[0]

X, y = [], []

for label, folder in [(0, "labeling_crops/man"), (1, "labeling_crops/woman")]:
    files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    print(f"Loading {len(files)} images from {folder}...")
    for f in files:
        emb = get_embedding(os.path.join(folder, f))
        X.append(emb)
        y.append(label)

X = np.array(X)
y = np.array(y)
print(f"\nTotal labeled examples: {len(X)} (man={sum(y==0)}, woman={sum(y==1)})")

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
scores = cross_val_score(clf, X, y, cv=5)
print(f"\n5-fold cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# still fit on everything for the final saved model
clf.fit(X, y)
joblib.dump(clf, "gender_classifier.pkl")
print("Saved trained classifier to gender_classifier.pkl")