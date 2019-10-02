import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from techprocess_tracking.TechProcessTracker import TechProcessTracker
from utils.VideoPlayback import VideoPlayback


class TechProcessTrackingThread(QThread):
    frameInfoReady = pyqtSignal(int, np.ndarray, float, int, int)

    def __init__(self, techProcessTracker: TechProcessTracker, videoSource, videoSourceDelayMs=None, parent=None):
        super(TechProcessTrackingThread, self).__init__(parent)
        self.techProcessTracker = techProcessTracker
        self.videoSource = videoSource
        self.videoSourceDelayMs = videoSourceDelayMs

    def __sleep(self):
        if self.videoSourceDelayMs and self.videoSourceDelayMs > 0:
            self.msleep(self.videoSourceDelayMs)

    def run(self):
        def _shareResults():
            pinsCount, pinsWithSolderCount = self.techProcessTracker.getStats()
            self.techProcessTracker.drawScene(frame, True)
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
            self.frameInfoReady.emit(pos, frame, msec, pinsCount, pinsWithSolderCount)

        video = VideoPlayback(self.videoSource)
        for pos, frame, msec in video.frames():
            self.techProcessTracker.track(pos, msec, frame)
            _shareResults()
            self.__sleep()
