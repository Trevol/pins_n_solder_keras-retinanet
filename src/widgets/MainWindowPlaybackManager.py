from detection.PinDetector import PickledDictionaryPinDetector
from segmentation.SceneSegmentation import CachedSceneSegmentation
from techprocess_tracking.TechProcessTracker import TechProcessTracker
from widgets import MainWindow
from widgets.threads.TechProcessTrackingThread import TechProcessTrackingThread


class MainWindowPlaybackManager():
    def __init__(self, parent):
        self.parent: MainWindow = parent
        self._trackingThread: TechProcessTrackingThread = None
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

    def startTracking(self):
        try:
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

    def __frameInfoReady(self, pos, frame, msec, pinsCount, pinsWithSolderCount, logRecord):
        self.parent.videoWidget.imshow(frame)
        self.parent.techProcessInfoWidget.setInfo(pos, msec, pinsCount, pinsWithSolderCount, logRecord)

    def startTechProcessTrackerThread(self):
        assert self._trackingThread is None

        videoSource = '/HDD_DATA/Computer_Vision_Task/Video_6.mp4'
        # videoSource = '/HDD_DATA/Computer_Vision_Task/Video_2.mp4'
        videoSourceDelayMs = 50

        # videoSource = 0
        # videoSourceDelayMs=-1 #no delay for camera feed

        def techProcessTrackerFactory():
            pinDetector = PickledDictionaryPinDetector('detection/csv_cache/data/detections_video6.pcl')
            sceneSegmentation = CachedSceneSegmentation(
                '/home/trevol/HDD_DATA/Computer_Vision_Task/frames_6/not_augmented_base_vgg16_more_images_25')

            # pinDetector = RetinanetPinDetector('modelWeights/retinanet_pins_inference.h5')
            # sceneSegmentation = UnetSceneSegmentation('modelWeights/unet_pins_25_0.000016_1.000000.hdf5')

            techProcessTracker = TechProcessTracker(pinDetector, sceneSegmentation)
            return techProcessTracker

        self._trackingThread = TechProcessTrackingThread(techProcessTrackerFactory, videoSource, videoSourceDelayMs)
        self._trackingThread.started.connect(self._trackingThreadStarted)
        self._trackingThread.finished.connect(self._trackingThreadFinished)

        self._trackingThread.frameInfoReady.connect(self.__frameInfoReady)
        self._trackingThread.start()
