from detection.PinDetector import PickledDictionaryPinDetector
from segmentation.SceneSegmentation import CachedSceneSegmentation
from techprocess_tracking.TechProcessTracker import TechProcessTracker
from widgets import MainWindow
from widgets.threads.TechProcessTrackingThread import TechProcessTrackingThread
from widgets.threads.VideoPlaybackThread import VideoPlaybackThread
from .videoTrackingConfig import videoSource, videoSourceDelayMs, techProcessTrackerFactory


class MainWindowPlaybackManager():
    def __init__(self, parent):
        self.parent: MainWindow = parent
        self._trackingThread: TechProcessTrackingThread = None
        self._playbackThread = None
        self.startPlayback()

    def startPlayback(self):
        assert self._playbackThread is None

        self._playbackThread = VideoPlaybackThread(videoSource, videoSourceDelayMs)
        self._playbackThread.frameReady.connect(self._playbackFrameReady)
        self._playbackThread.start()

    def _playbackFrameReady(self, pos, frame, msec):
        self.parent.videoWidget.imshow(frame)

    def stopPlayback(self):
        if self._playbackThread is not None:
            self._playbackThread.finish()
            self._playbackThread = None


    def startOrStop(self):
        if self._trackingThread:
            # started!
            self.stopTracking()
        else:
            self.startTracking()

    def stopTracking(self):
        self.parent.startStopButton.setDisabled(True)
        self._trackingThread.finish()
        self.startPlayback()

    def startTracking(self):
        try:
            self.stopPlayback()
            self.parent.clearTrackingInfo()
            self.parent.startStopButton.setDisabled(True)
            self.startTechProcessTrackerThread()
        except:
            self._trackingThread = None
            self.parent.startStopButton.setEnabled(True)
            raise

    def _trackingThreadStarted(self):
        self.parent.startStopButton.setText('Stop')
        self.parent.startStopButton.setEnabled(True)

    def _trackingThreadFinished(self):
        self._trackingThread = None
        self.parent.startStopButton.setText('Start')
        self.parent.startStopButton.setEnabled(True)

    def shutdown(self, waitMsecs=500):
        if self._trackingThread:
            self._trackingThread.finish()
            if waitMsecs > 0:
                self._trackingThread.wait(waitMsecs)
        if self._playbackThread:
            self._playbackThread.finish()
            if waitMsecs > 0:
                self._playbackThread.wait(waitMsecs)

    def _frameInfoReady(self, pos, frame, msec, pinsCount, pinsWithSolderCount, logRecord):
        self.parent.videoWidget.imshow(frame)
        self.parent.techProcessInfoWidget.setInfo(pos, msec, pinsCount, pinsWithSolderCount, logRecord)

    def startTechProcessTrackerThread(self):
        assert self._trackingThread is None

        self._trackingThread = TechProcessTrackingThread(techProcessTrackerFactory, videoSource, videoSourceDelayMs)
        self._trackingThread.started.connect(self._trackingThreadStarted)
        self._trackingThread.finished.connect(self._trackingThreadFinished)

        self._trackingThread.frameInfoReady.connect(self._frameInfoReady)
        self._trackingThread.start()
