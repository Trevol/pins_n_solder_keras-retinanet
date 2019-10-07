from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, qApp, QVBoxLayout, QPushButton, QLayout

from detection.PinDetector import PickledDictionaryPinDetector, RetinanetPinDetector
from segmentation.SceneSegmentation import CachedSceneSegmentation, UnetSceneSegmentation
from techprocess_tracking.TechProcessTracker import TechProcessTracker
from widgets.TechProcessTrackingThread import TechProcessTrackingThread
from .TechProcessInfoWidget import TechProcessInfoWidget
from .VideoWidget import VideoWidget


class MainWindow(QMainWindow):
    class ThreadManager():
        def __init__(self, parent):
            self.parent: MainWindow = parent
            self._trackingThread: TechProcessTrackingThread = None
            self._playbackThread = None
            self.trackingStopped = True

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

        def shutdown(self, waitForCompletion=False, msecs=500):
            if self._trackingThread:
                self._trackingThread.finish()
                self._trackingThread.wait(msecs)
            if self._playbackThread:
                self._trackingThread.finish()
                self._trackingThread.wait(msecs)

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

    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()
        self.threadManager = self.ThreadManager(self)

    def initUI(self):
        self.setWindowTitle("Process")
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        self.initMainLayout(centralWidget)

    def initMainLayout(self, centralWidget: QWidget):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.videoWidget = VideoWidget()
        layout.addWidget(self.videoWidget, stretch=1)

        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 6, 0, 0)

        self.startStopButton = QPushButton('Start')
        self.startStopButton.clicked.connect(self.startOrStop)
        vbox.addWidget(self.startStopButton, stretch=0, alignment=Qt.AlignLeft)

        self.techProcessInfoWidget = TechProcessInfoWidget()
        vbox.addWidget(self.techProcessInfoWidget)

        layout.addLayout(vbox)

        centralWidget.setLayout(layout)

    def keyPressEvent(self, keyEvent: QtGui.QKeyEvent):
        if keyEvent.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.hide()
        self.threadManager.shutdown(waitForCompletion=True)
        self.close()

    def startOrStop(self):
        self.threadManager.startOrStop()

    def clearTrackingInfo(self):
        self.videoWidget.imshow(None)
        self.techProcessInfoWidget.clearInfo()
