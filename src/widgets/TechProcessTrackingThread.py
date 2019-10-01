from PyQt5.QtCore import QThread

from techprocess_tracking.TechProcessTracker import TechProcessTracker
from utils.VideoPlayback import VideoPlayback


class TechProcessTrackingThread(QThread):
    def __init__(self, techProcessTracker: TechProcessTracker, videoSource, videoSourceDelay=None, parent=None):
        super(TechProcessTrackingThread, self).__init__(parent)
        self.techProcessTracker = techProcessTracker
        self.videoSource = videoSource
        self.videoSourceDelay = videoSourceDelay

    def run(self):
        video = VideoPlayback(self.videoSource)
        for pos, frame, msec in video.frames():
            self.techProcessTracker.track(pos, msec, frame)
