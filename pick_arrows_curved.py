import cv2
import json

img = cv2.imread("clean_background.jpg")
h, w = img.shape[:2]

scale = 0.6
display_img = cv2.resize(img, None, fx=scale, fy=scale)

arrows = []
current_points = []

def redraw():
    temp = display_img.copy()
    if len(current_points) >= 2:
        pts = [(int(x*scale), int(y*scale)) for x, y in current_points]
        cv2.polylines(temp, [np_array(pts)], False, (0, 255, 0), 2)
    for x, y in current_points:
        cv2.circle(temp, (int(x*scale), int(y*scale)), 4, (0, 0, 255), -1)
    cv2.imshow("Place curved arrows", temp)

def np_array(pts):
    import numpy as np
    return np.array(pts, dtype=int)

def click_event(event, x, y, flags, param):
    global current_points
    if event == cv2.EVENT_LBUTTONDOWN:
        orig_x, orig_y = int(x / scale), int(y / scale)
        current_points.append((orig_x, orig_y))
        redraw()

cv2.imshow("Place curved arrows", display_img)
cv2.setMouseCallback("Place curved arrows", click_event)
print("Click multiple points to trace a curved path (start to end, in order).")
print("Press 'n' to finish this arrow and name it, then start the next one.")
print("Press 'q' when done placing all arrows.")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('n') and len(current_points) >= 2:
        name = input("Name this arrow: ")
        arrows.append({"name": name, "points": current_points})
        current_points = []
        redraw()
        print(f"Arrow '{name}' saved. Click points for the next arrow, or 'q' to finish.")
    elif key == ord('q'):
        break

cv2.destroyAllWindows()

with open("arrows.json", "w") as f:
    json.dump(arrows, f, indent=2)

print(f"\nSaved {len(arrows)} arrows to arrows.json")
