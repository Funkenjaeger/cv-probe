from __future__ import print_function
import cv2
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

img_0_0 = cv2.imread('img/20230409/img_0_0.png')
img_1_0 = cv2.imread('img/20230409/img_1_0.png')
img_0_neg1 = cv2.imread('img/20230409/img_0_neg1.png')
img_1_neg1 = cv2.imread('img/20230409/img_1_neg1.png')
height = img_0_0.shape[0]


v1 = np.asarray([[239, 452],
                 [243, 201],
                 [486, 203],
                 [486, 452]], dtype=np.int32)

v2 = np.asarray([[203, 476],
                 [210, 177],
                 [506, 181],
                 [507, 479]], dtype=np.int32)


def find_disparity(img1, img2, v):
    ref = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    xmin, xmax = np.min(v[:, 0]), np.max(v[:, 0])
    ymin, ymax = np.min(v[:, 1]), np.max(v[:, 1])
    ref = ref[ymin:ymax+1, xmin:xmax+1]
    cmp = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    tm = cv2.matchTemplate(img1_gray, ref, method=cv2.TM_SQDIFF)
    minlocref = cv2.minMaxLoc(tm)[2]  # (x, y)
    tm = cv2.matchTemplate(cmp, ref, method=cv2.TM_SQDIFF)
    minloccmp = cv2.minMaxLoc(tm)[2]  # (x, y)
    # TODO: this assumes camera moved in -Y direction; otherwise need to mirror the mask
    return minlocref[1] - minloccmp[1]


d0 = find_disparity(img_0_0, img_1_0, v1)
d1 = find_disparity(img_0_neg1, img_1_neg1, v2)

print(f'Disparity at Z0: {d0}')
print(f'Disparity at Z-2: {d1}')

delta_z = -2
z0 = delta_z * d1 / (d0 - d1)
print(f'z0 est: {z0:.2f}, f/d est: {z0*d0*2/2:.1f}')

print('Done')
