import cv2
import numpy as np

table1_img = [[224, 285], [358, 357], [524, 198], [450, 162]]
table1_floor = [[0, 0], [1.2, 0], [1.2, 2.4], [0, 2.4]]

table2_img = [[476, 145], [549, 170], [584, 133], [552, 111]]
table2_floor = [[0, 3.6], [1.2, 3.6], [1.2, 6.0], [0, 6.0]]

table3_img = [[50, 197], [100, 236], [317, 149], [259, 121]]
table3_floor = [[-2.0, 0], [-0.8, 0], [-0.8, 2.4], [-2.0, 2.4]]

image_points = np.array(table1_img + table2_img + table3_img, dtype=np.float32)
floor_points = np.array(table1_floor + table2_floor + table3_floor, dtype=np.float32)

H, status = cv2.findHomography(image_points, floor_points)
print("Homography matrix:")
print(H)
np.save("homography.npy", H)
print("Saved updated homography.npy (12 points)")