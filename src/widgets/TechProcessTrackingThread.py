import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from techprocess_tracking.TechProcesLogRecord import TechProcesLogRecord
from techprocess_tracking.TechProcessTracker import TechProcessTracker
from utils.VideoPlayback import VideoPlayback


class TechProcessTrackingThread(QThread):
    frameInfoReady = pyqtSignal(int, np.ndarray, float, int, int, object)

    def __init__(self, techProcessTrackerFactory, videoSource, videoSourceDelayMs=None, parent=None):
        super(TechProcessTrackingThread, self).__init__(parent)
        self.techProcessTrackerFactory = techProcessTrackerFactory
        self.videoSource = videoSource
        self.videoSourceDelayMs = videoSourceDelayMs

    def _sleep(self):
        if self.videoSourceDelayMs and self.videoSourceDelayMs > 0:
            self.msleep(self.videoSourceDelayMs)

    def _execProcessTracking(self):
        def emitResults():
            pinsCount, pinsWithSolderCount = techProcessTracker.getStats()
            techProcessTracker.drawScene(frame, True)
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
            self.frameInfoReady.emit(pos, frame, msec, pinsCount, pinsWithSolderCount, log)

        video = VideoPlayback(self.videoSource)
        techProcessTracker = self.techProcessTrackerFactory()
        for pos, frame, msec in video.frames():
            log = techProcessTracker.track(pos, msec, frame)
            emitResults()
            self._sleep()

    def run(self):
        try:
            self._execProcessTracking()
        except BaseException as e:
            print(type(e), e)
