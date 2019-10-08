import cv2
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np

from utils.VideoPlayback import VideoPlayback


class VideoPlaybackThread(QThread):
    frameReady = pyqtSignal(int, np.ndarray, float)

    def __init__(self, videoSource, videoSourceDelayMs=None):
        super(VideoPlaybackThread, self).__init__()
        self._finishRequired = False
        self.videoSource = videoSource
        self.videoSourceDelayMs = videoSourceDelayMs

    def _sleep(self):
        if self.videoSourceDelayMs and self.videoSourceDelayMs > 0:
            self.msleep(self.videoSourceDelayMs)

    def _play(self):
        def emitResults():
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
            self.frameReady.emit(pos, frame, msec)

        video = VideoPlayback(self.videoSource)
        try:
            for pos, frame, msec in video.frames():
                if self._finishRequired:
                    break
                emitResults()
                self._sleep()
                if self._finishRequired:
                    break
        finally:
            video.release()

    def run(self):
        try:
            self._play()
        except BaseException as e:
            print(type(e), e)

    def finish(self):
        self._finishRequired = True
