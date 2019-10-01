import time
from time import time, sleep

from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QHBoxLayout, QVBoxLayout, QLabel, QVBoxLayout, \
    QHBoxLayout, QWidget, QDesktopWidget, qApp
import sys
import cv2

from utils.VideoPlayback import VideoPlayback
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

    def imshow(self, img):
        self.videoWidget.imshow(img)


class VideoThread(QThread):
    def __init__(self, videoFile, parent=None):
        super(VideoThread, self).__init__(parent)
        self.videoFile = videoFile

    def run(self):
        video = VideoPlayback(self.videoFile)
        try:
            for pos, frame, _ in video.frames():
                if pos % 100 == 0:
                    print(pos)
            print(pos, 'Done!')
        finally:
            video.release()


def main():
    app = QApplication(sys.argv)
    w = Window()

    thread = VideoThread('/HDD_DATA/Computer_Vision_Task/Video_6.mp4')
    thread.start()

    sys.exit(app.exec())


main()
