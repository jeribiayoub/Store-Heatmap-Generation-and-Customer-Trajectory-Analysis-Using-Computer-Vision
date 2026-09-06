import cv2
import numpy as np

blueprint = cv2.imread("blueprint.png")
sample_points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        sample_points.append((x, y))
        color = blueprint[y, x].tolist()
        print(f"Sampled point ({x},{y}) -> BGR {color}")
        cv2.circle(display, (x, y), 4, (0, 0, 255), -1)
        cv2.imshow("Click on empty floor areas", display)

display = blueprint.copy()
cv2.imshow("Click on empty floor areas", display)
cv2.setMouseCallback("Click on empty floor areas", click_event)
print("Click 8-10 points on clearly EMPTY FLOOR areas (not shelves/tables), spread across the whole plan.")
print("Press 'q' when done.")
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

# compute average floor color + tolerance from your samples
colors = np.array([blueprint[y, x] for x, y in sample_points])
mean_color = colors.mean(axis=0)
std_color = colors.std(axis=0)
print(f"\nMean floor color (BGR): {mean_color}")
print(f"Std deviation: {std_color}")

# build a tolerance range around the sampled floor color
tolerance = np.maximum(std_color * 3, 15)  # at least 15, or 3x natural variation
lower = np.clip(mean_color - tolerance, 0, 255)
upper = np.clip(mean_color + tolerance, 0, 255)

floor_mask = cv2.inRange(blueprint, lower, upper)

# clean up small noise/holes
kernel = np.ones((5, 5), np.uint8)
floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel)
floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel)

cv2.imwrite("floor_mask.png", floor_mask)

# preview: floor area tinted green, so you can visually check accuracy
preview = blueprint.copy()
preview[floor_mask > 0] = (0.5 * preview[floor_mask > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
cv2.imwrite("floor_mask_preview.jpg", preview)

print("\nSaved floor_mask.png and floor_mask_preview.jpg")