import numpy as np

H = np.load("homography.npy")

test_points = [
    (224, 285),  # near-left corner of table (should map to ~[0,0])
    (358, 357),  # near-right corner (should map to ~[1.2, 0])
    (524, 198),  # far-right corner (should map to ~[1.2, 2.4])
    (450, 162),  # far-left corner (should map to ~[0, 2.4])
]

for pt in test_points:
    p = np.array([pt[0], pt[1], 1.0])
    fp = H @ p
    x, y = fp[0]/fp[2], fp[1]/fp[2]
    print(f"{pt} -> ({x:.2f}, {y:.2f}) meters")