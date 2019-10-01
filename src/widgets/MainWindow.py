from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, qApp

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

    def keyPressEvent(self, keyEvent: QtGui.QKeyEvent):
        if keyEvent.key() == Qt.Key_Escape:
            qApp.exit()
