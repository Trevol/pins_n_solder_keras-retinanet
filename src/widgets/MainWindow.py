from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton

from widgets.MainWindowPlaybackManager import MainWindowPlaybackManager
from .TechProcessInfoWidget import TechProcessInfoWidget
from .VideoWidget import VideoWidget


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()
        self.playbackManager = MainWindowPlaybackManager(self)

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
        self.playbackManager.shutdown()
        self.close()

    def startOrStop(self):
        self.playbackManager.startOrStop()

    def clearTrackingInfo(self):
        self.videoWidget.imshow(None)
        self.techProcessInfoWidget.clearInfo()
