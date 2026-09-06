import cv2
import numpy as np
import json

# load data
with open("positions_stitched_smoothed.json") as f:
    positions = json.load(f)

blueprint = cv2.imread("blueprint.png")
bp_h, bp_w = blueprint.shape[:2]

# build a density map: one bin per pixel of the blueprint
heatmap = np.zeros((bp_h, bp_w), dtype=np.float32)

for p in positions:
    x, y = int(p["x"]), int(p["y"])
    if 0 <= x < bp_w and 0 <= y < bp_h:
        heatmap[y, x] += 1

print("Raw max density (before smoothing):", heatmap.max())

# smooth it so it looks like a continuous heatmap, not scattered dots
heatmap_smooth = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=20, sigmaY=20)

# normalize to 0-255 for visualization
heatmap_norm = cv2.normalize(heatmap_smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# colorize (JET: blue=low density, red=high density)
heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

# mask out areas with essentially zero density, so empty floor doesn't get tinted blue
mask = heatmap_norm > 10
overlay = blueprint.copy()
overlay[mask] = cv2.addWeighted(blueprint, 0.4, heatmap_color, 0.6, 0)[mask]

cv2.imwrite("heatmap_raw.jpg", heatmap_norm)
cv2.imwrite("heatmap_overlay.jpg", overlay)
print("Saved heatmap_raw.jpg and heatmap_overlay.jpg")