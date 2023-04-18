import cv2
import numpy as np
import glob
import json
from tkinter import filedialog
import os.path

imgs = []
dir = filedialog.askdirectory(mustexist=True)
images = glob.glob(os.path.join(dir, '*.png'))

for filename in images:
    imgs.append(cv2.imread(filename))

print(f'Image size: {imgs[0].shape}')

CHECKERBOARD = (6, 9)
scale = (5.54/6.0, 5.57/6.0) # measured w/ calipers

criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Vectors for 3D & 2D points
threedpoints = []
twodpoints = []

#  3D points real world coordinates
objectp3d = np.zeros((1, CHECKERBOARD[0]
                      * CHECKERBOARD[1],
                      3), np.float32)
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                               0:CHECKERBOARD[1]].T.reshape(-1, 2)
objectp3d[0, :, 0] *= scale[0]
objectp3d[0, :, 1] *= scale[1]

prev_img_shape = None

for img in imgs:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_FAST_CHECK +
        cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        threedpoints.append(objectp3d)

        # Refining pixel coordinates
        # for given 2d points.
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)

        twodpoints.append(corners2)

        img = cv2.drawChessboardCorners(img,
                                        CHECKERBOARD,
                                        corners2, ret)
        cv2.imshow('Input image', img)
        cv2.waitKey(0)

cv2.destroyAllWindows()
h, w = img.shape[:2]

ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
    threedpoints, twodpoints, gray.shape[::-1], None, None)

# Displaying required output
print(" Camera matrix:")
print(matrix)

print("\n Distortion coefficient:")
print(distortion)

print("\n Rotation Vectors:")
print(r_vecs)

print("\n Translation Vectors:")
print(t_vecs)

mean_error = 0
for i in range(len(threedpoints)):
    imgpoints2, _ = cv2.projectPoints(threedpoints[i], r_vecs[i], t_vecs[i],
                                      matrix, distortion)
    error = cv2.norm(twodpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print("total error: {}".format(mean_error/len(threedpoints)))

h, w = imgs[0].shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion,
                                                  (w, h), 1, (w, h))
for img in imgs:
    ud = cv2.undistort(img, matrix, distortion, None, newcameramtx)
    #x, y, w, h = roi
    #ud = ud[y:y+h, x:x+w]
    cv2.imshow('Undistorted', ud)
    cv2.waitKey(0)

cv2.destroyAllWindows()

with filedialog.asksaveasfile(mode='w') as file:
    json.dump({'matrix': matrix.tolist(),
               'distortion': distortion.tolist()},
              file)
