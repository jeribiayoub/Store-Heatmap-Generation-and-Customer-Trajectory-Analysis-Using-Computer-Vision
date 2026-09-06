import cv2
import numpy as np

world_points = np.array([
    [0, 0], [1.2, 0], [1.2, 2.4], [0, 2.4],
    [0, 3.6], [1.2, 3.6], [1.2, 6.0], [0, 6.0],
    [-2.0, 0], [-0.8, 0], [-0.8, 2.4], [-2.0, 2.4],
], dtype=np.float32)

blueprint_points = np.array([
    [342, 663], [388, 663], [386, 463], [344, 463],
    [343, 349], [388, 350], [386, 169], [342, 170],
    [167, 662], [215, 663], [215, 465], [168, 464],
], dtype=np.float32)

M, inliers = cv2.estimateAffine2D(world_points, blueprint_points)
print("Affine matrix (world -> blueprint):")
print(M)
np.save("world_to_blueprint.npy", M)
print("Saved updated world_to_blueprint.npy (12 points)")

print("\nSanity check:")
for wp, bp in zip(world_points, blueprint_points):
    p = np.array([wp[0], wp[1], 1.0])
    proj = M @ p
    print(f"world {tuple(wp)} -> predicted {proj.round(1)}  (actual {tuple(bp)})")