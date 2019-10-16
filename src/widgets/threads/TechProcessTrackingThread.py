import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from utils.VideoPlayback import VideoPlayback


class TechProcessTrackingThread(QThread):
    frameInfoReady = pyqtSignal(int, np.ndarray, float, int, int, object)

    def __init__(self, techProcessTracker, videoSource, videoSourceDelayMs=None, parent=None):
        super(TechProcessTrackingThread, self).__init__(parent)
        self._finishRequired = False
        self.techProcessTracker = techProcessTracker
        self.videoSource = videoSource
        self.videoSourceDelayMs = videoSourceDelayMs

    def _sleep(self):
        if self.videoSourceDelayMs and self.videoSourceDelayMs > 0:
            self.msleep(self.videoSourceDelayMs)

    def _execProcessTracking(self):
        def emitResults():
            pinsCount, pinsWithSolderCount = self.techProcessTracker.getStats()
            self.techProcessTracker.drawScene(frame, True)
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
            self.frameInfoReady.emit(pos, frame, msec, pinsCount, pinsWithSolderCount, log)

        video = VideoPlayback(self.videoSource)
        try:
            for pos, frame, msec in video.frames(4300):
                log = self.techProcessTracker.track(pos, msec, frame)
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
            self._execProcessTracking()
        except BaseException as e:
            print(type(e), e)

    def finish(self):
        self._finishRequired = True
