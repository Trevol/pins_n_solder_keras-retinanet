from widgets import MainWindow
from widgets.threads.TechProcessTrackingThread import TechProcessTrackingThread
from widgets.threads.VideoPlaybackThread import VideoPlaybackThread
from .videoTrackingConfig import videoSource, videoSourceDelayMs, techProcessTrackerFactory


class MainWindowPlaybackManager():
    class Tracking:
        def __init__(self, manager):
            self.manager: MainWindowPlaybackManager = manager
            self._thread: TechProcessTrackingThread = None

        def shutdown(self, waitMsecs):
            if self._thread is None:
                return
            self._thread.finish()
            if waitMsecs > 0:
                self._thread.wait(waitMsecs)
            self._thread = None

        def started(self):
            return self._thread is not None

        def stop(self):
            self.manager.mainWindow.startStopButton.setDisabled(True)
            self._thread.finish()
            self.manager.playback.start()

        def start(self):
            try:
                self.manager.playback.stop()
                self.manager.mainWindow.clearTrackingInfo()
                self.manager.mainWindow.startStopButton.setDisabled(True)
                self.startThread()
            except:
                self._thread = None
                self.manager.mainWindow.startStopButton.setEnabled(True)
                raise

        def _threadStarted(self):
            self.manager.mainWindow.startStopButton.setText('Stop')
            self.manager.mainWindow.startStopButton.setEnabled(True)

        def _threadFinished(self):
            self._thread = None
            self.manager.mainWindow.startStopButton.setText('Start')
            self.manager.mainWindow.startStopButton.setEnabled(True)

        def _frameInfoReady(self, pos, frame, msec, pinsCount, pinsWithSolderCount, logRecord):
            self.manager.mainWindow.videoWidget.imshow(frame)
            self.manager.mainWindow.techProcessInfoWidget.setInfo(pos, msec, pinsCount, pinsWithSolderCount, logRecord)

        def startThread(self):
            assert self._thread is None

            self._thread = TechProcessTrackingThread(techProcessTrackerFactory, videoSource, videoSourceDelayMs)
            self._thread.started.connect(self._threadStarted)
            self._thread.finished.connect(self._threadFinished)

            self._thread.frameInfoReady.connect(self._frameInfoReady)
            self._thread.start()

    class Playback:
        def __init__(self, manager):
            self.manager: MainWindowPlaybackManager = manager
            self._thread = None

        def shutdown(self, waitMsecs):
            if self._thread is None:
                return
            self._thread.finish()
            if waitMsecs > 0:
                self._thread.wait(waitMsecs)
            self._thread = None

        def start(self):
            assert self._thread is None

            self._thread = VideoPlaybackThread(videoSource, videoSourceDelayMs)
            self._thread.frameReady.connect(self._frameReady)
            self._thread.start()

        def _frameReady(self, pos, frame, msec):
            self.manager.mainWindow.videoWidget.imshow(frame)

        def stop(self):
            if self._thread is not None:
                self._thread.finish()
                self._thread = None

    def __init__(self, mainWindow):
        self.mainWindow: MainWindow = mainWindow
        self.tracking = self.Tracking(self)
        self.playback = self.Playback(self)
        self.playback.start()

    def startOrStopTracking(self):
        if self.tracking.started():
            self.tracking.stop()
        else:
            self.tracking.start()

    def shutdown(self, waitMsecs=500):
        self.tracking.shutdown(waitMsecs)
        self.playback.shutdown(waitMsecs)
