from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QLabel, QScrollArea, QSizePolicy, QBoxLayout, QHBoxLayout
import numpy as np


class VideoWidget(QWidget):
    def __init__(self):
        super(VideoWidget, self).__init__()
        self.__initUI__()

    def __initUI__(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        scroll = QScrollArea(self)
        layout.addWidget(scroll)

        self.imageLabel = QLabel()
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)
        scroll.setWidget(self.imageLabel)

    def imshow(self, img):
        if img is None:
            self.imageLabel.clear()
            self.imageLabel.resize(1, 1)
            self.imageLabel.setVisible(False)
        else:
            self.imageLabel.setVisible(True)
            self.imageLabel.setPixmap(_get_pixmap(img))
            self.imageLabel.resize(img.shape[1], img.shape[0])


def _get_pixmap(image):
    if image is None:
        return None
    # w, h, format = _get_image_size_n_format(image)
    # qimage = QImage(image.data, w, h, image.strides[0], format)
    qimage = ndarray_to_qimage(image)
    return QPixmap.fromImage(qimage)


def ndarray_to_qimage(arr: np.ndarray):
    """
    Convert NumPy array to QImage object
    credits: https://github.com/PierreRaybaut/PythonQwt/blob/master/qwt/toqimage.py

    :param numpy.array arr: NumPy array
    :return: QImage object
    """
    # https://gist.githubusercontent.com/smex/5287589/raw/toQImage.py
    if arr is None:
        return QImage()
    if len(arr.shape) not in (2, 3):
        raise NotImplementedError("Unsupported array shape %r" % arr.shape)
    data = arr.data
    ny, nx = arr.shape[:2]
    stride = arr.strides[0]  # bytes per line
    color_dim = None
    if len(arr.shape) == 3:
        color_dim = arr.shape[2]
    if arr.dtype == np.uint8:
        if color_dim is None:
            qimage = QImage(data, nx, ny, stride, QImage.Format_Indexed8)
            #            qimage.setColorTable([qRgb(i, i, i) for i in range(256)])
            qimage.setColorCount(256)
        elif color_dim == 3:
            qimage = QImage(data, nx, ny, stride, QImage.Format_RGB888)
        elif color_dim == 4:
            qimage = QImage(data, nx, ny, stride, QImage.Format_ARGB32)
        else:
            raise TypeError("Invalid third axis dimension (%r)" % color_dim)
    elif arr.dtype == np.uint32:
        qimage = QImage(data, nx, ny, stride, QImage.Format_ARGB32)
    else:
        raise NotImplementedError("Unsupported array data type %r" % arr.dtype)
    return qimage
