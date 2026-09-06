import cv2
import numpy as np

image_points = np.array([
    [224, 285],  # near-left
    [358, 357],  # near-right
    [524, 198],  # far-right
    [450, 162],  # far-left
], dtype=np.float32)

W = 1.2  # table width, meters (near-left to near-right)
L = 2.4  # table length, meters (near to far)

floor_points = np.array([
    [0, 0],   # near-left  -> origin
    [W, 0],   # near-right
    [W, L],   # far-right
    [0, L],   # far-left
], dtype=np.float32)

H, status = cv2.findHomography(image_points, floor_points)
print("Homography matrix:")
print(H)
np.save("homography.npy", H)  # save it, we'll reuse this in every future step

cap = cv2.VideoCapture("data/3.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
ret, frame = cap.read()
cap.release()

scale = 150  # pixels per meter for the preview
out_w, out_h = int(8 * scale), int(8 * scale)

S = np.array([[scale, 0, 250], [0, scale, 50], [0, 0, 1]])  # scale + shift so it's centered in view
H_view = S @ H

warped = cv2.warpPerspective(frame, H_view, (out_w, out_h))
cv2.imwrite("warped_check2.jpg", warped)