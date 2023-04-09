import cv2
import numpy as np
import lineprocessing


class Target:

    _pts_picked = []
    _vertices_origin = []
    _vertices = []
    _edges = []
    _pos = None
    _bb_center = None
    _flat_img_sel = None
    _blur_size = None
    _canny_thr1 = None
    _canny_thr2 = None
    _edged_img = None

    def __init__(self, img, origin_viewport: tuple[float, float, float],
                 flat: str = 'BGR',
                 blur: int = 3,
                 thr1: int = 25,
                 thr2: int = 100):
        self.img = img.copy()
        self._origin_viewport = origin_viewport
        self._pos = origin_viewport
        self._flat_img_sel = flat
        self._blur_size = blur
        self._canny_thr1 = thr1
        self._canny_thr2 = thr2
        self._edge_detect()

    @property
    def vertices(self):
        return self._vertices

    @property
    def edges(self):
        return self._edges

    def _edge_detect(self):
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        match self._flat_img_sel:
            case 'gray':
                img2process = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            case 'Hsv' | 'hue':
                img2process = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)[:, :, 0]
            case 'hSv' | 'saturation':
                img2process = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)[:, :, 1]
            case 'hsV' | 'value':
                img2process = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)[:, :, 2]
            case 'BGR' | 'bgr':
                img2process = self.img.copy()
            case default:
                img2process = None

        blurred = cv2.GaussianBlur(img2process,
                                   (self._blur_size, self._blur_size), 0)
        self._edged_img = cv2.Canny(blurred, self._canny_thr1, self._canny_thr2)

    def pick_point(self, pt, delete=False):
        (x, y) = pt

        # If we've already got a set of vertices, then the user must be trying
        # to modify one.  Recall the vertices as picked points, so they can be
        # manipulated
        if len(self._pts_picked) == 0 and len(self._vertices) == 4:
            self._pts_picked = [(x, y) for (x, y, theta) in self._vertices]

        # Check for proximity to an existing picked point
        d = [((x-x0) ** 2 + (y-y0) ** 2) < 20 ** 2 for (x0, y0) in
             self._pts_picked]
        if any(d):
            if delete:
                self._pts_picked.pop(d.index(True))  # Delete point
            else:
                self._pts_picked[d.index(True)] = (x, y)  # Replace point
        elif len(self._pts_picked) < 4:
            self._pts_picked.append((x, y))
        self._recalculate()

    def params(self, flat: str = None,
                 blur: int = None,
                 thr1: int = None,
                 thr2: int = None):
        if flat is not None:
            self._flat_img_sel = flat

        if blur is not None:
            # must be an odd number
            if np.mod(blur, 2) == 1:
                self._blur_size = blur
            else:
                print('Gaussian blur kernel size must be an odd number '
                      f'(got: {blur})')

        if thr1 is not None:
            self._canny_thr1 = thr1

        if thr2 is not None:
            self._canny_thr2 = thr2

        self._edge_detect()
        self._recalculate()

    def _recalculate(self, update=True):
        # Sort points by vector angle relative to bounding box center
        if self._pts_picked is None or len(self._pts_picked) == 0:
            return []
        x, y, w, h = cv2.boundingRect(np.asarray(self._pts_picked))
        xc, yc = x + w / 2, y + h / 2
        theta = [np.arctan2(y - yc, x - xc) for (x, y) in self._pts_picked]
        pts_picked = [pt for _, pt in sorted(zip(theta, self._pts_picked))]

        # Find edges from picked points
        edges = []
        for i in range(0, len(pts_picked)):
            if len(pts_picked) > 1:
                rho, theta = lineprocessing.find_line(self._edged_img,
                                                      pts_picked[i - 1],
                                                      pts_picked[i])
                if rho:
                    edges.append((rho, theta))

        # Find vertices from edges
        vertices = []
        for i in range(0, len(edges)):
            if len(edges) > 1:
                x, y, theta = lineprocessing.find_intersection(
                    edges[i - 1],
                    edges[i])
                if x:
                    vertices.append((x, y, theta))

        if update:
            self._vertices = vertices.copy()
            self._edges = edges.copy()
            self._bb_center = (xc, yc)
            if len(self._vertices) == 4:
                # Remove picked points once we have a full set of 4 vertices
                self._pts_picked = []
                if self._pos == self._origin_viewport:
                    self._vertices_origin = self._vertices.copy()
            else:
                self._pts_picked = pts_picked.copy()

        return vertices, edges

    @staticmethod
    def _point_text(img,  x, y, color=(255, 255, 0)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.line(img, (x - 10, y), (x + 10, y), color, 2)
        cv2.line(img, (x, y - 10), (x, y + 10), color, 2)
        cv2.putText(img, f'{x},{y}', (x + 10, y + 10), font, 0.5, color, 1)

    def annotate_image(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        img_sel_text = f'Flat: {self._flat_img_sel}'
        cv2.putText(img, img_sel_text, (10, 20), font, 0.5, (0, 255, 0), 1)
        cv2.putText(img, f'Blur size: {self._blur_size}', (10, 40), font,
                    0.5, (0, 255, 0), 1)

        for pt in self._pts_picked:
            (x, y) = pt
            Target._point_text(img, x, y)

        for line in self._edges:
            (rho, theta) = line
            lineprocessing.draw_line(img, rho, theta)

        for vertex in self._vertices:
            (x, y, theta) = vertex
            r = 10
            color = (255, 0, 0)
            cv2.circle(img, (x, y), radius=r, color=color, thickness=2)
            (xc, yc) = self._bb_center
            sgnx, sgny = np.sign(xc - x), np.sign(yc - y)
            x2, y2 = int(x + sgnx * r), int(y + sgny * r)
            cv2.line(img, (x, y), (x2, y),
                     color=color, thickness=3)
            cv2.line(img, (x, y), (x, y2),
                     color=color, thickness=3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f'{theta * 180 / np.pi:.1f}'
            ((dx, dy), _) = cv2.getTextSize(text, font, 0.5, 1)
            xt = x2 if sgnx > 0 else x2 - dx
            yt = y2 if sgny < 0 else y2 + dy
            cv2.putText(img, text, (xt, yt), font, 0.5, color, 1)

    def frame(self):
        img = self.img.copy()
        self.annotate_image(img)
        return img

    def frame_edged(self):
        img = self._edged_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self.annotate_image(img)
        return img

    def translate(self, img, pos: tuple[float, float, float]):
        if len(self._vertices_origin) == 0:
            return False
        self.img = img
        self._pos = pos
        self._edge_detect()
        (x0, y0, z0) = self._origin_viewport
        (x, y, z) = pos
        dx, dy = x-x0, y-y0
        # TODO: f/d shouldn't be hard coded
        # TODO: deal with dz (scale)
        f_by_d = 1320/2 # ~1320 @ full resolution, here we're scaled to 50%
        scale = z0/z
        yc, xc = img.shape[0]/2, img.shape[1]/2
        disparity = (f_by_d * dx / z, f_by_d * dy / z)

        vertices_expected = []
        for i in range(0, len(self._vertices)):
            (xv, yv, tv) = self._vertices_origin[i]
            xv = int((xv - xc) * scale + xc + disparity[0])
            yv = int((yv - yc) * scale + yc + disparity[1])
            vertices_expected.append((xv, yv, tv))
        self._pts_picked = [(x, y) for (x, y, _) in vertices_expected]
        v, e = self._recalculate(update=False)
        if len(v) == 4 and len(vertices_expected) == 4:
            self._recalculate()  # Now update for real
            print(f'Successfully translated from {self._origin_viewport} to '
                  f'{pos}, disparity {disparity}')
            errs = []
            for i in range(0, len(vertices_expected)):
                (x, y, _) = v[i]
                (xe, ye, _) = vertices_expected[i]
                err = np.round(np.sqrt((x-xe)**2 + (y-ye)**2), decimals=1)
                errs.append(err)
            print(f'Vertex position errors (pixels): {errs} '
                  f'({np.mean(errs):.1f} avg)')
            return True
        else:
            print(f'Failed to translate from {self._origin_viewport} to '
                  f'{pos}, disparity {disparity} - only got {len(v)} vertices')
            self._vertices = vertices_expected.copy()
            # TODO: handle case where some vertices are out of frame
            # TODO: handle case where some vertices are in frame but not found
            # Probably just use the expected values for each missing vertex and
            # its 2 associated lines

            return False
