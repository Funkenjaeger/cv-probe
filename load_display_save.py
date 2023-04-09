from __future__ import print_function
import argparse
import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
import rerun as rr
import lineprocessing
from target import Target

#rr.init("my data", spawn=True)

image1 = None
loaded = False
edged = None
img_display = None


def try_get_trackbar(name, win):
    try:
        return cv2.getTrackbarPos(name, win)

    except cv2.error as e:
        if e.code == -215:
            return 0
        else:
            raise e


def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        delete = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
        target.pick_point((x, y), delete=delete)
        if try_get_trackbar('Display img', 'win'):
            cv2.imshow('win', target.frame_edged())
        else:
            cv2.imshow('win', target.frame())


def reset_go(val=0):
    global target

    distance = cv2.getTrackbarPos('Image sel', 'win')
    if distance == 3:
        distance = 4
    image1 = cv2.imread('img\\cvtest_' + str(distance) + '_0.png')
    image1 = cv2.resize(image1,
                        (int(image1.shape[1] / 2), int(image1.shape[0] / 2)))

    target = Target(image1, origin_viewport=(0, 0, 5.978 + float(distance)))

    go()


def go(val=0):
    global loaded, target

    if not loaded:
        return

    blur = try_get_trackbar('Blur size', 'win') * 2 + 1
    thr1 = cv2.getTrackbarPos('Canny thr1', 'win')
    thr2 = cv2.getTrackbarPos('Canny thr2', 'win')
    display_img = try_get_trackbar('Display img', 'win')
    flat_img_sel = try_get_trackbar('Flat image', 'win')
    flat = ['gray', 'hue', 'saturation', 'value', 'BGR'][flat_img_sel]

    target.params(flat=flat, blur=blur, thr1=thr1, thr2=thr2)

    if display_img:
        cv2.imshow('win', target.frame_edged())
    else:
        cv2.imshow('win', target.frame())


def translate(val=0):
    match try_get_trackbar('translation', 'win'):
        case 0:
            x = 0
            s = '0'
        case 1:
            x = 0.25
            s = '0.25'
        case 2:
            x = 1.0
            s = '1'
        case default:
            return

    distance = try_get_trackbar('Image sel', 'win')
    if distance == 3:
        distance = 4

    img = cv2.imread('img\\cvtest_' + str(distance) + '_' + s + '.png')
    img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
    pos = (0, x, 5.978 + float(distance))
    target.translate(img, pos)
    if try_get_trackbar('Display img', 'win'):
        cv2.imshow('win', target.frame_edged())
    else:
        cv2.imshow('win', target.frame())


cv2.namedWindow('win', cv2.WINDOW_AUTOSIZE)

distance = 2
image1 = cv2.imread('img\\cvtest_' + str(distance) + '_0.png')
image1 = cv2.resize(image1,
                    (int(image1.shape[1] / 2), int(image1.shape[0] / 2)))
target = Target(image1, origin_viewport=(0, 0, 5.978 + float(distance)))

cv2.createTrackbar('Image sel', 'win', distance, 3, translate)
cv2.createTrackbar('Flat image', 'win', 4, 4, go)
cv2.createTrackbar('Blur size', 'win', 3, 10, go)
cv2.createTrackbar('Canny thr1', 'win', 25, 255, go)
cv2.createTrackbar('Canny thr2', 'win', 100, 255, go)
cv2.createTrackbar('Display img', 'win', 0, 1, go)
cv2.createTrackbar('translation', 'win', 0, 2, translate)


loaded = True
go()
cv2.setMouseCallback('win', mouse_click)

cv2.waitKey(0)
cv2.destroyAllWindows()
