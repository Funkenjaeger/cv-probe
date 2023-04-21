from __future__ import print_function
import cv2
from target import Target
import linuxcnc
import time
import depth
import numpy as np
import json

image1 = None
loaded = False
edged = None
img_display = None

cnc_s = linuxcnc.stat()
cnc_c = linuxcnc.command()

cnc_s.poll()
print(f'Actual position: {cnc_s.actual_position}')


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
        target.pick_point((x*2, y*2), delete=delete)
        if try_get_trackbar('Display img', 'win'):
            img = target.frame_edged()
        else:
            img = target.frame()
        img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
        cv2.imshow('win', img)


def reset_go(val=0):
    global target, image1

    for i in range(0, 10):
        cam.read()
        time.sleep(0.1)
    result, img0 = cam.read()
    target.img = img0.copy()
    target._pts_picked = []
    target._vertices = []
    target._edges = []

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
        img = target.frame_edged()
    else:
        img = target.frame()
    img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
    cv2.imshow('win', img)


def translate(val=0):
    t = try_get_trackbar('translation', 'win')
    if t is not None:
        if t == 0:
            x = 0
            s = '0'
        elif t == 1:
            x = 0.25
            s = '0.25'
        elif t == 2:
            x = 1.0
            s = '1'
        else:
            return
    else:
        return

    distance = try_get_trackbar('Image sel', 'win')
    if distance == 3:
        distance = 4

    img = cv2.imread('img/cvtest_' + str(distance) + '_' + s + '.png')
    img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
    pos = (0, x, 5.978 + float(distance))
    target.translate(img, pos)
    if try_get_trackbar('Display img', 'win'):
        cv2.imshow('win', target.frame_edged())
    else:
        cv2.imshow('win', target.frame())


def ok_for_mdi():
    cnc_s.poll()
    homed = True
    for joint in cnc_s.joint:
        homed = homed and ((not joint['enabled']) or (joint['homed'] != 0))
    # TODO: check actual enum for STATE_ESTOP
    return cnc_s.estop==-1 and cnc_s.enabled and homed and (cnc_s.interp_state == linuxcnc.INTERP_IDLE)


def get_pos():
    cnc_s.poll()
    p = cnc_s.actual_position
    return p[0:3]


cv2.namedWindow('win', cv2.WINDOW_AUTOSIZE)

# cam = cv2.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw, width=2560, height=1440 ! videoconvert ! video/x-raw,format=BGR ! appsink")
cam = cv2.VideoCapture(-1)
if not cam.isOpened():
    print("Failed to open camera")
    exit()
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # turn the autofocus on
# result, image = cam.read()
# time.sleep(1)
result, image = cam.read()
var = cv2.Laplacian(image, cv2.CV_64F).var()
print(f'Laplacian variance: {var}')
# TODO: if the variance sucks (100+ is good, ~50 is passable, <20 is badly blurry) try to make it autofocus
# TODO: once satisfied with the focus, turn off autofocus
if result:
    image1 = image
else:
    print("Failed to get image")
    exit()

distance = 2
# cv2.imread('img/cvtest_' + str(distance) + '_0.png')
# image1 = cv2.resize(image1, (int(image1.shape[1] / 2), int(image1.shape[0] / 2)))
(x, y, z) = get_pos()
target = Target(image1, origin_viewport=(0, 0, 5.978 + float(distance)))

cv2.createTrackbar('Image sel', 'win', distance, 3, translate)
cv2.createTrackbar('Flat image', 'win', 4, 4, go)
cv2.createTrackbar('Blur size', 'win', 3, 10, go)
cv2.createTrackbar('Canny thr1', 'win', 20, 255, go)
cv2.createTrackbar('Canny thr2', 'win', 75, 255, go)
cv2.createTrackbar('Display img', 'win', 0, 1, go)
cv2.createTrackbar('translation', 'win', 0, 2, translate)
cv2.createTrackbar('reset', 'win', 0, 1, reset_go)


loaded = True
go()
cv2.setMouseCallback('win', mouse_click)

with open('camera-cal_anker.json') as file:
    obj = json.load(file)
matrix = np.asarray(obj.get('matrix', None))
distortion = np.asarray(obj.get('distortion', None))
h, w = image1.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion,
                                                  (w, h), 1, (w, h))

t_update = time.time()
while True:
    key = cv2.pollKey()
    if key > 0:  # Ref asciitable.com for decimal values
        print(f'Got key {key}')
    if cv2.getWindowProperty('win', cv2.WND_PROP_VISIBLE) < 1:
        break
    if target.is_found:
        while True:
            print('Ready to find depth, press spacebar to continue')
            key = cv2.waitKey(0)
            if key == 32:  # space
                if ok_for_mdi():
                    cnc_c.mode(linuxcnc.MODE_MDI)
                    (vx0, vy0, _) = target.vertices[0]  # TODO: pick vertices with smartness
                    (vx1, vy1, _) = target.vertices[1]
                    l = min(vx0, vx1) - 10
                    r = max(vx0, vx1) + 10

                    cnc_c.mdi(f'G53 G90 G0 Z{0:0.5f}')
                    rv = cnc_c.wait_complete(5)
                    time.sleep(2)
                    (x0, y0, z0) = get_pos()
                    for i in range(0, 2):
                        cam.read()
                        time.sleep(0.1)
                    result, img0 = cam.read()
                    img0 = cv2.undistort(img0, matrix, distortion, None,
                                       newcameramtx)

                    cnc_c.mdi(f'G53 G90 G0 X{x0 + 2.0:0.5f} Z{z0:0.5f}')
                    rv = cnc_c.wait_complete(5)
                    if rv != 1:
                        print('MDI command timed out')
                        break
                    for i in range(0, 2):
                        cam.read()
                        time.sleep(0.1)
                    result, img1 = cam.read()
                    img1 = cv2.undistort(img1, matrix, distortion, None,
                                         newcameramtx)
                    v = np.asarray([[x, y] for (x, y, _) in target.vertices], dtype=np.int32)
                    disparity = depth.estimate_disparity(img0, img1, v)

                    '''cnc_c.mdi(f'G53 G90 G0 X{x0:0.5f} Z{z0 - 2.0:0.5f}')
                    rv = cnc_c.wait_complete(5)
                    for i in range(0, 10):
                        cam.read()
                        time.sleep(0.1)
                    result, img0 = cam.read()

                    cnc_c.mdi(f'G53 G90 G0 X{x0 + 2.0:0.5f} Z{z0 - 2.0:0.5f}')
                    rv = cnc_c.wait_complete(5)
                    for i in range(0, 10):
                        cam.read()
                        time.sleep(0.1)
                    result, img1 = cam.read()
                    # DEBUG
                    x1, y1, z1 = 0, 0, -2
                    dx, dy = 0, 0
                    # TODO: f/d shouldn't be hard coded
                    # TODO: deal with dz (scale)
                    f_by_d = 1320 / 2  # ~1320 @ full resolution, here we're scaled to 50%
                    scale = (z0 + 11) / (z1 + 11)
                    yc, xc = img1.shape[0] / 2, img1.shape[1] / 2
                    disparity = (f_by_d * dx / (z1+11), f_by_d * dy / (z1+11))

                    vertices_translated = []
                    for i in range(0, len(target.vertices)):
                        (xv, yv, _) = target.vertices[i]
                        xv = int((xv - xc) * scale + xc + disparity[0])
                        yv = int((yv - yc) * scale + yc + disparity[1])
                        vertices_translated.append([xv, yv])
                    # END DEBUG
                    vertices_translated = np.asarray(vertices_translated, dtype=np.int32)
                    disparity1 = depth.estimate_disparity(img0, img1, vertices_translated)
                    z0_est, f_by_d_est = depth.estimate_depth(disparity0, disparity1, 2, -2)'''

                    cnc_c.mdi(f'G53 G90 G0 X{x0:0.5f} Z{z0:0.5f}')
                    rv = cnc_c.wait_complete(5)
                    z0_est = matrix[0, 0] * 2.0 / disparity

                    print(f'Disparity at Z0: {disparity}')
                    print(f'Estimated Z0: {z0_est:.2f}')
                    # print('Successfully collected all 2  images, press a thing to continue')
                    # cv2.waitKey(0)
                    break
                else:
                    print('Machine not ready for MDI')
    '''if (time.time() - t_update) > 0.1:
        result, image = cam.read()
        if result:
            image1 = image
            cnc_s.poll()
            pos = cnc_s.actual_position
            target.translate(image, pos[0:3])
        else:
            print("Failed to get image")'''

cv2.destroyAllWindows()
