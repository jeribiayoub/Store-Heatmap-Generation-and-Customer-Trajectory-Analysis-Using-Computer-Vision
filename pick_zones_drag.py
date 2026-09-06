import cv2
import json

img = cv2.imread("clean_background.jpg")
h, w = img.shape[:2]

scale = 0.6  # adjust if still too big/small for your screen (e.g. 0.5 or 0.7)
display_img = cv2.resize(img, None, fx=scale, fy=scale)

print("Draw a rectangle by clicking and dragging.")
print("Press ENTER or SPACE to confirm each box, then draw the next one.")
print("Press ESC when you're done drawing all zones.")

rois = cv2.selectROIs("Draw zones - drag boxes, ENTER to confirm each, ESC when done", display_img)
cv2.destroyAllWindows()

zones = []
for i, (x, y, box_w, box_h) in enumerate(rois):
    # convert back to original full-resolution coordinates
    orig_x, orig_y = int(x / scale), int(y / scale)
    orig_w, orig_h = int(box_w / scale), int(box_h / scale)

    name = input(f"Name for zone {i+1} (box at x={orig_x}, y={orig_y}): ")
    zones.append({
        "name": name,
        "x1": orig_x, "y1": orig_y,
        "x2": orig_x + orig_w, "y2": orig_y + orig_h,
    })

with open("zones.json", "w") as f:
    json.dump(zones, f, indent=2)

print(f"\nSaved {len(zones)} zones to zones.json")
for z in zones:
    print(z)