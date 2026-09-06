import cv2
import numpy as np

# video pixel points (8 points, 2 islands)
video_points = np.array([
    [371, 469], [557, 510], [1030, 266], [938, 244],  # island 1
    [962, 599], [1238, 669], [1509, 359], [1357, 331],  # island 2
], dtype=np.float32)

# matching blueprint pixel points, same order
blueprint_points = np.array([
    [415, 643], [505, 644], [505, 273], [415, 274],  # island 1
    [764, 643], [855, 643], [854, 274], [765, 273],  # island 2
], dtype=np.float32)

H_direct, status = cv2.findHomography(video_points, blueprint_points)
print("Direct video->blueprint homography:")
print(H_direct)
np.save("video_to_blueprint.npy", H_direct)
print("Saved video_to_blueprint.npy")

print("\nSanity check:")
for vp, bp in zip(video_points, blueprint_points):
    p = np.array([vp[0], vp[1], 1.0])
    proj = H_direct @ p
    x, y = proj[0]/proj[2], proj[1]/proj[2]
    print(f"video {tuple(vp)} -> predicted ({x:.1f}, {y:.1f})  (actual {tuple(bp)})")