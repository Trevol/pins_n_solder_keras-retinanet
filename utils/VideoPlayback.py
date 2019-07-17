import cv2

from .KbdKeys import KbdKeys


class VideoPlayback:
    def __init__(self, videoPath):
        self.cap = cv2.VideoCapture(videoPath)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def __currentPos(self):
        assert self.cap
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def setPos(self, pos):
        assert self.cap
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    def backward(self, numFrames=1):
        assert self.cap
        newPos = self.__currentPos() - numFrames - 1
        if newPos < 0:
            newPos = 0
        self.setPos(newPos)

    def frames(self):
        assert self.cap
        while True:
            frame = self.readFrame()
            if frame is None:
                break
            yield self.__currentPos() - 1, frame

    def readFrame(self, pos=None):
        if pos is not None:
            self.setPos(pos)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def framesCount(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def handleKey(self):
        key = cv2.waitKeyEx()
        if key == KbdKeys.L_ARROW:
            self.backward()
        return key