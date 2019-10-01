from time import time, sleep

from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QHBoxLayout, QVBoxLayout, QLabel, QVBoxLayout, \
    QHBoxLayout, QWidget, QDesktopWidget, qApp
import sys
import cv2

from widgets.VideoWidget import VideoWidget


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def keyPressEvent(self, keyEvent: QtGui.QKeyEvent):
        if keyEvent.key() == Qt.Key_Escape:
            self.close()

    def initUI(self):
        self.setGeometry(50, 50, 500, 400)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        videoWidget = VideoWidget()
        layout.addWidget(videoWidget)
        self.setLayout(layout)
        self.videoWidget = videoWidget

        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Window()

    img = cv2.imread('segmentation/train/dataset/image/f_3439_229266.67_229.27.jpg')
    assert img is not None
    w.videoWidget.imshow(img)

    sys.exit(app.exec())
