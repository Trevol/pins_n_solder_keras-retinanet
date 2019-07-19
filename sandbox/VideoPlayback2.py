import cv2
from utils import leftClip
from .VideoController2 import VideoController2


class VideoPlayback2:
    def __init__(self, videoPath, initialFrameDelay=1, autoplayInitially=True):
        self.cap = cv2.VideoCapture(videoPath)
        self.frameDelay = initialFrameDelay
        self.autoPlay = autoplayInitially
        self._controller = VideoController2(self)

    @property
    def manualPlay(self):
        return not self.autoPlay

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

    def changeFrameDelay(self, direction):
        # manage change value depending of current frameDelay magnitude
        if self.frameDelay > 500:
            step = 20
        elif self.frameDelay > 100:
            step = 10
        elif self.frameDelay > 20:
            step = 5
        else:
            step = 1

        self.frameDelay += step * direction
        self.frameDelay = leftClip(self.frameDelay, 1)

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

    def handleAction(self):
        return self._controller.handleAction()

    def play(self, onFrameReady=None, onStateChange=None):
        def defaultFrameReady(frame, framePos, playback):
            cv2.imshow('Video', frame)

        def defaultStateChange(frameDelay, autoPlay, framePos, playback):
            autoplayLabel = 'ON' if autoPlay else 'OFF'
            stateTitle = f'Video (Pos: {framePos}, FrameDelay: {frameDelay}, Autoplay: {autoplayLabel})'
            cv2.setWindowTitle('Video', stateTitle)

        if onFrameReady is None and onStateChange is None:
            onStateChange = defaultStateChange
        onFrameReady = onFrameReady or defaultFrameReady

        for pos, frame in self.frames():
            onFrameReady(frame, pos, self)
            onStateChange(self.frameDelay, self.autoPlay, pos, self)
            while True:
                read, stop, changed, key = self.handleAction()  # enter in user action handling
                if changed:
                    onStateChange(self.frameDelay, self.autoPlay, pos, self)
                if read:
                    break
                if stop:
                    return
