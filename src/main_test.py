import time
from time import time, sleep

from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QHBoxLayout, QVBoxLayout, QLabel, QVBoxLayout, \
    QHBoxLayout, QWidget, QDesktopWidget, qApp, QLineEdit
import sys
import cv2
import numpy as np

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
        layout.setContentsMargins(1, 1, 1, 1)
        self.setLayout(layout)

        self.videoWidget = VideoWidget()
        layout.addWidget(self.videoWidget)

        self.label = QLineEdit()
        self.label.setReadOnly(True)
        layout.insertWidget(0, self.label)

    def imshow(self, img):
        self.videoWidget.imshow(img)


class VideoThread(QThread):
    stateChanged = pyqtSignal(int, np.ndarray, float, bool)

    def __init__(self, videoFile, parent=None):
        super(VideoThread, self).__init__(parent)
        self.videoFile = videoFile

    def run(self):
        video = VideoPlayback(self.videoFile)
        try:
            pos = -1
            frame = None
            mSec = -1.
            for pos, frame, mSec in video.frames():
                self.msleep(1000 // 20)
                self.stateChanged.emit(pos, frame, mSec, False)
            self.stateChanged.emit(pos, frame, mSec, True)
        finally:
            video.release()


def main():
    app = QApplication(sys.argv)
    wnd = Window()

    def onThreadStateChanged(pos, frame, mSec, done):
        wnd.label.setText(f'{pos} {mSec:.2f} ms Done: {done}')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=frame)
        wnd.imshow(frame)

    thread = VideoThread('/HDD_DATA/Computer_Vision_Task/Video_6.mp4')
    thread.stateChanged.connect(onThreadStateChanged, Qt.QueuedConnection)

    wnd.show()
    thread.start()

    sys.exit(app.exec())


main()
