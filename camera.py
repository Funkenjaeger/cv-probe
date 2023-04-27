import cv2
import numpy as np
import json
import queue
import threading
import time


class BufferlessVideoCapture:

    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
        self.frame_q = queue.Queue()
        self.cmd_q = queue.Queue()
        self.response_q = queue.Queue()

        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            if not self.cmd_q.empty():
                cmd = self.cmd_q.get_nowait()
                prop = cmd[1]
                if cmd[0] == 'set':
                    val = cmd[2]
                    self.cap.set(prop, val)
                elif cmd[0] == 'get':
                    val = self.cap.get(prop)
                    self.response_q.put(val)
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.frame_q.empty():
                try:
                    self.frame_q.get_nowait()
                except queue.Empty:
                    pass
            self.frame_q.put(frame)

    def read(self):
        return self.frame_q.get()

    def set(self, prop, val):
        self.cmd_q.put(('set', prop, val))

    def get(self, prop):
        self.cmd_q.put(('get', prop))
        val = self.response_q.get(block=True, timeout=1)
        return val


class Camera:

    def __init__(self, config, index=0):
        self._config = config
        self.frame = None
        self._in_focus = False
        self._cam = BufferlessVideoCapture(index)
        self._cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.get_frame()
        self._var = 0
        '''if self.in_focus:
            self._cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)'''

    def get_frame(self):
        self.frame = self._cam.read()
        self._var = cv2.Laplacian(self.frame, cv2.CV_64F).var()
        self._in_focus = self._var > 100
        af = self._cam.get(cv2.CAP_PROP_AUTOFOCUS) > 0
        cv2.putText(self.frame, f'Var: {self._var:.1f}, AF lock: {not af}', fontScale=1., org=(100, 100),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 255, 0))
        return self.frame


    @property
    def in_focus(self):
        return self._in_focus

    @property
    def var(self):
        return self._var

    def lock_focus(self, optimize=False):
        t_start = time.time()
        self._cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        f_min = self._config.get('camera', 'f_min')
        f_max = self._config.get('camera', 'f_max')
        f_start = self._config.get('camera', 'f_default')
        f_threshold = self._config.get('camera', 'f_threshold')
        n_window = self._config.get('camera', 'n_window')
        f_step = 4

        var = 0
        f = f_start
        f_tried = {}
        while f_min <= f <= f_max:
            if f not in f_tried:
                self._cam.set(cv2.CAP_PROP_FOCUS, f)

                # Watch and wait for focus to stabilize
                vs = []
                while True:
                    self.get_frame()
                    if len(vs) == 0 or self._var != vs[-1]:
                        vs.append(self._var)
                    if len(vs) >= n_window and max(vs[-n_window:]) - min(vs[-n_window:]) < 5.0:
                        break
                f_tried[f] = self._var

                print(f'Focus {f}, var={self._var:.1f}')

                if self._var > f_threshold and not optimize:
                    self._config.set('camera', 'f_default', f)
                    print(f'Focused at f={f}, var={self._var:.1f} elapsed time: {time.time() - t_start:.1f} sec')
                    return True

                if self._var < var and var > 100:
                    if abs(f_step) == 1:
                        f_final = f - f_step
                        self._cam.set(cv2.CAP_PROP_FOCUS, f_final)
                        self._config.set('camera', 'f_default', f_final)
                        print(f'Focused at f={f_final}, var={var:.1f}, elapsed time: {time.time() - t_start:.1f} sec')
                        return True
                    f_sign = np.sign(f_step)
                    if f == f_start + f_step:
                        f_step *= -1
                    else:
                        f_step = int(max(1, np.abs(f_step/2)) * -f_sign)

                var = self._var
            else:
                var = f_tried[f]
            f += f_step

        self._cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        return False
