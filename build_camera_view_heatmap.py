import cv2
import numpy as np
import json

with open("raw_boxes.json") as f:
    raw_boxes = json.load(f)

background = cv2.imread("clean_background.jpg")
h, w = background.shape[:2]

# accumulate raw foot points directly in camera pixel coordinates
heatmap = np.zeros((h, w), dtype=np.float32)

for b in raw_boxes:
    foot_x = (b["x1"] + b["x2"]) / 2
    foot_y = b["y2"]
    x, y = int(foot_x), int(foot_y)
    if 0 <= x < w and 0 <= y < h:
        heatmap[y, x] += 1

print("Raw max density:", heatmap.max())

# smooth
heatmap_smooth = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=25, sigmaY=25)

# normalize and colorize
norm = cv2.normalize(heatmap_smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

# blend onto the real photo
overlay = cv2.addWeighted(background, 0.5, color, 0.5, 0)

cv2.imwrite("heatmap_camera_view.jpg", overlay)
print("Saved heatmap_camera_view.jpg")