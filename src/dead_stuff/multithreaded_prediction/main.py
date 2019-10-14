from contextlib import contextmanager
from threading import Lock

import cv2
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication

import numpy as np

from detection.RetinanetPinDetector import RetinanetPinDetector
from models.ModelsContext import ModelsContext

from segmentation.UnetSceneSegmentation import UnetSceneSegmentation

class TestThread(QThread):
    lock = Lock()

    def __init__(self, testFrame, id):
        super(TestThread, self).__init__()
        self.id = id
        self.testFrame = testFrame

    def run(self):
        try:
            with self.lock:
                print('Thread ', int(self.id), ' started!!!')
            with ModelsContext() as ctx:
                for _ in range(30):
                    boxes, detections = ctx.getDetector().detect(self.testFrame, 366, .85)
                    ctx.getSegmentation().getSegmentationMap(self.testFrame, 366)
            with self.lock:
                print('Thread ', int(self.id), ' finished!!!')
        except Exception as e:
            print('ERROR!', e)
            raise


def test_case_1():
    """
    create in main thread and predict in different sequential threads
    """

    framePath = 'f_0366_24400.00_24.40.jpg'
    frame = cv2.imread(framePath)

    thread1 = TestThread(frame, 11111)
    thread2 = TestThread(frame, 22222)
    thread3 = TestThread(frame, 33333)
    thread4 = TestThread(frame, 44444)
    thread5 = TestThread(frame, 55555)

    thread1.start()
    thread1.wait()
    thread2.start()
    thread2.wait()
    thread3.start()
    thread3.wait()
    thread4.start()
    thread4.wait()
    thread5.start()
    thread5.wait()


def test_case_2():
    """
    create in main thread and predict in different sequential threads
    """

    framePath = 'f_0366_24400.00_24.40.jpg'
    frame = cv2.imread(framePath)

    thread1 = TestThread(frame, 11111)
    thread2 = TestThread(frame, 22222)
    thread3 = TestThread(frame, 33333)
    thread4 = TestThread(frame, 44444)
    thread5 = TestThread(frame, 55555)

    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()

    thread1.wait()
    thread2.wait()
    thread3.wait()
    thread4.wait()
    thread5.wait()

def main():
    app = QApplication([])

    test_case_2()
    test_case_1()


main()
