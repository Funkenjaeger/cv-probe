import cv2
import numpy as np
import json
import Queue
import threading
import time


class BufferlessVideoCapture:

    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        self.q = Queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except Queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


class Camera:

    def __init__(self, index=0):
        self.frame = None
        self._cam = BufferlessVideoCapture(index)
        if not self._cam.isOpened():
            print("Failed to open camera")
            return None
        self._cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self._cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
        self._cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.get_frame()
        if self.in_focus:
            self._cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)


    def get_frame(self):
        self.frame = self._cam.read()
        return self.frame

    @property
    def in_focus(self):
        if self.frame is None:
            return False
        var = cv2.Laplacian(self.frame, cv2.CV_64F).var()
        return var > 100

    def lock_focus(self):
        if self.in_focus:
            self._cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self._cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        for i in range(0,10):
            time.sleep(0.5)
            self.get_frame()
            if self.in_focus:
                self._cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                return True
        return False
