from __future__ import print_function
import cv2
import numpy as np


def find_line(edgedimg, pt1, pt2, margin=10):
    mask = edgedimg.copy()
    mask.fill(0)
    cv2.line(mask, pt1, pt2, color=255, thickness=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2*margin+1, 2*margin+1),
                                       (margin, margin))
    mask = cv2.dilate(mask, kernel)

    img = edgedimg.copy()
    img = cv2.bitwise_and(img, mask)

    threshold = 0.5 * np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
    lines = cv2.HoughLinesWithAccumulator(img, rho=1, theta=0.2 * np.pi / 180,
                                          threshold=int(threshold))

    if lines is None:
        # Try again with even lower threshold
        lines = cv2.HoughLinesWithAccumulator(img, rho=1,
                                              theta=0.5 * np.pi / 180,
                                              threshold=int(threshold * 0.5))
        if lines is None:
            # Still nothing -> give up
            return None, None

    img = cv2.cvtColor(edgedimg.copy(), cv2.COLOR_GRAY2BGR)
    scores = lines[0:3, 0, 2]
    smin = np.min(scores)
    srng = np.max(scores) - smin
    for i in range(lines.shape[0]).__reversed__():
        [rho, theta, score] = lines[i, 0, :]
        if len(lines) == 1 or srng < 1:
            c = 255
        else:
            c = int(255.0 * (score - smin) / srng)
        color = (0, 255, 0) if i == 0 else (0, 0, c)
        draw_line(img, rho, theta, color)

    [rho, theta, _] = lines[0, 0, :]  # Return highest-scoring line
    return rho, theta


def draw_line(img, rho, theta, color=(0, 0, 255)):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * -b)
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * -b)
    y2 = int(y0 - 1000 * a)
    cv2.line(img, (x1, y1), (x2, y2), color, 1)


def find_intersection(line1, line2):
    (rho1, theta1) = line1
    (rho2, theta2) = line2
    c1, c2 = np.cos(theta1), np.cos(theta2)
    s1, s2 = np.sin(theta1), np.sin(theta2)
    det = c1 * s2 - s1 * c2
    if det == 0.0:
        return None, None, None
    x = int((s2 * rho1 - s1 * rho2) / det)
    y = int((-c2 * rho1 + c1 * rho2) / det)
    theta = np.mod(np.pi - (theta2-theta1), np.pi)
    return x, y, theta
