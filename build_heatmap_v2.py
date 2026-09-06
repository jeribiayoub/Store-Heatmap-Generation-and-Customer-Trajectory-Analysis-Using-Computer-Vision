import cv2
import numpy as np
import json

with open("positions_stitched_smoothed.json") as f:
    positions = json.load(f)

blueprint = cv2.imread("blueprint.png")
floor_mask = cv2.imread("floor_mask.png", cv2.IMREAD_GRAYSCALE)
bp_h, bp_w = blueprint.shape[:2]

# accumulate raw density
heatmap = np.zeros((bp_h, bp_w), dtype=np.float32)
for p in positions:
    x, y = int(p["x"]), int(p["y"])
    if 0 <= x < bp_w and 0 <= y < bp_h:
        heatmap[y, x] += 1

# smooth
heatmap_smooth = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=20, sigmaY=20)

# normalize using ONLY the floor area's max, so shelf pixels (always 0) don't skew the scale
floor_values = heatmap_smooth[floor_mask > 0]
vmax = floor_values.max() if floor_values.max() > 0 else 1.0

# 0 density -> 0 (pure blue in JET), vmax density -> 255 (pure red in JET)
norm = np.clip(heatmap_smooth / vmax * 255, 0, 255).astype(np.uint8)
color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)  # JET: blue (low) -> cyan -> yellow -> red (high)

# start from the full blueprint, but REPLACE every floor pixel with its color
# (not blended - so even zero-traffic floor becomes solid blue, as requested)
result = blueprint.copy()
result[floor_mask > 0] = color[floor_mask > 0]

cv2.imwrite("heatmap_full_coverage.jpg", result)
print("Saved heatmap_full_coverage.jpg")
print(f"Max density value used for scaling: {vmax:.1f}")