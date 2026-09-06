import cv2

img = cv2.imread("blueprint.png")
if img is None:
    raise RuntimeError("Couldn't load blueprint.png - check it's in the project folder")

# scale down just for display, so it fits your screen
scale = 0.6  # adjust if still too big/small (e.g. 0.5 or 0.7)
display_img = cv2.resize(img, None, fx=scale, fy=scale)

points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        orig_x, orig_y = int(x / scale), int(y / scale)
        points.append((orig_x, orig_y))
        print(f"Point {len(points)}: ({orig_x}, {orig_y})")
        cv2.circle(display_img, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(display_img, str(len(points)), (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Pick blueprint points", display_img)

cv2.imshow("Pick blueprint points", display_img)
cv2.setMouseCallback("Pick blueprint points", click_event)
print("Click the matching border points, same order as the video points.")
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

print("\nFinal pixel points (original resolution):", points)