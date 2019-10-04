from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, qApp, QVBoxLayout, QPushButton, QLayout

from detection.PinDetector import PickledDictionaryPinDetector, RetinanetPinDetector
from segmentation.SceneSegmentation import CachedSceneSegmentation, UnetSceneSegmentation
from techprocess_tracking.TechProcessTracker import TechProcessTracker
from widgets.TechProcessTrackingThread import TechProcessTrackingThread
from .TechProcessInfoWidget import TechProcessInfoWidget
from .VideoWidget import VideoWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()
        self.__thread = None
        self.startTechProcessTrackerThread()

    def initUI(self):
        self.setWindowTitle("Process")
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        self.initMainLayout(centralWidget)

    def initMainLayout(self, centralWidget: QWidget):
        # TODO: add start/stop buttons
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
        # TODO: stop thread
        pass

    def startOrStop(self):
        print('startOrStop')

    def __frameInfoReady(self, pos, frame, msec, pinsCount, pinsWithSolderCount, logRecord):
        self.videoWidget.imshow(frame)
        self.techProcessInfoWidget.setInfo(pos, msec, pinsCount, pinsWithSolderCount, logRecord)

    def startTechProcessTrackerThread(self):
        assert self.__thread is None

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

        self.__thread = TechProcessTrackingThread(techProcessTrackerFactory, videoSource, videoSourceDelayMs)
        self.__thread.frameInfoReady.connect(self.__frameInfoReady)
        self.__thread.start()
