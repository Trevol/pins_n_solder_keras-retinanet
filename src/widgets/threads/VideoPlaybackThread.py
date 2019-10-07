from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np

class VideoPlaybackThread(QThread):
    frameReady = pyqtSignal(int, np.ndarray, float, int, int, object)
    raise NotImplementedError()