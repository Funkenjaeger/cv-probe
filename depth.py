from __future__ import print_function
import cv2
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


def estimate_disparity_OLD(img1, img2, l, r, f_by_d_guess=660, max_sane_z=12):
    # TODO: take delta X and delta Z as inputs!
    dx = 2
    min_sane_disparity = dx * f_by_d_guess / max_sane_z
    gray1 = cv2.cvtColor(img1[:, l:r + 1, :], cv2.COLOR_BGR2GRAY).astype(float)
    gray1 -= np.mean(gray1)
    gray2 = cv2.cvtColor(img2[:, l:r + 1, :], cv2.COLOR_BGR2GRAY).astype(float)
    gray2 -= np.mean(gray2)
    xcorr_gray = np.max(scipy.signal.fftconvolve(gray1, gray2[::-1, ::-1], 'same'), 1)
    midpoint = int(img1.shape[0] / 2)
    # plt.figure(2)
    # plt.plot(xcorr_gray, label='raw')
    xcorr_gray[0:int(midpoint + min_sane_disparity)] = 0
    # TODO: this assumes camera moved in -Y direction; otherwise need to mirror the mask
    # plt.plot(xcorr_gray, label='masked')
    # plt.legend()
    # plt.show()
    return np.argmax(xcorr_gray) - midpoint


def estimate_disparity(img1, img2, v):
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
    # TODO: this assumes camera moved in +X direction; otherwise need to mirror the mask
    return minloccmp[0] - minlocref[0]


def estimate_depth(d0, d1, dx, dz):
    z0 = dz * d1 / (d0 - d1)
    f_by_d = z0 * d0 / dx
    return z0, f_by_d
