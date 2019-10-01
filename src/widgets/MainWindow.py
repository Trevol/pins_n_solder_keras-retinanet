from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, qApp

from detection.PinDetector import PickledDictionaryPinDetector
from segmentation.SceneSegmentation import CachedSceneSegmentation
from techprocess_tracking.TechProcessTracker import TechProcessTracker
from .TechProcessInfoWidget import TechProcessInfoWidget
from .VideoWidget import VideoWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("My Window")
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        self.initMainLayout(centralWidget)

    def initMainLayout(self, centralWidget: QWidget):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(VideoWidget())
        layout.addWidget(TechProcessInfoWidget())
        centralWidget.setLayout(layout)

    def initTracker(self):
        pinDetector, sceneSegmentation = createServices(pclFile, segmentationCacheDir)
        self.techProcessTracker = TechProcessTracker(pinDetector, sceneSegmentation)


    def keyPressEvent(self, keyEvent: QtGui.QKeyEvent):
        if keyEvent.key() == Qt.Key_Escape:
            qApp.exit()

def files():
    yield ('/HDD_DATA/Computer_Vision_Task/Video_6.mp4',
           'detection/csv_cache/data/detections_video6.pcl',
           '/home/trevol/HDD_DATA/Computer_Vision_Task/frames_6/not_augmented_base_vgg16_more_images_25')

    # yield ('/HDD_DATA/Computer_Vision_Task/Video_2.mp4',
    #        '../../csv_cache/data/detections_video2.pcl')


def createServices(pclFile, segmentationCacheDir):
    pinDetector = PickledDictionaryPinDetector(pclFile)
    # pinDetector = RetinanetPinDetector('modelWeights/retinanet_pins_inference.h5')

    sceneSegmentation = CachedSceneSegmentation(segmentationCacheDir)
    # sceneSegmentation = UnetSceneSegmentation('modelWeights/unet_pins_25_0.000016_1.000000.hdf5')
    return pinDetector, sceneSegmentation
