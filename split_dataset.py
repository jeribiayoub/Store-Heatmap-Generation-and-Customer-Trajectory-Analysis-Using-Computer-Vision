import os
import shutil
import random

IMAGES_DIR = "training_frames"
LABELS_DIR = "labels_export"  # <-- put your extracted .txt files here first

random.seed(42)

image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")]
random.shuffle(image_files)

split_idx = int(len(image_files) * 0.85)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

for split_name, files in [("train", train_files), ("val", val_files)]:
    for img_file in files:
        base = os.path.splitext(img_file)[0]
        label_file = base + ".txt"

        src_img = os.path.join(IMAGES_DIR, img_file)
        dst_img = os.path.join("training_data/images", split_name, img_file)
        shutil.copy(src_img, dst_img)

        src_label = os.path.join(LABELS_DIR, label_file)
        dst_label = os.path.join("training_data/labels", split_name, label_file)
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
        else:
            print(f"Warning: no label found for {img_file}")

print(f"Train: {len(train_files)} images, Val: {len(val_files)} images")