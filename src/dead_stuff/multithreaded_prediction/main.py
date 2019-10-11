from threading import Lock

import cv2
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication

import numpy as np

from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
from detection.Box import Box

import tensorflow as tf
import keras.backend as K

class RetinanetPinDetector:
    def __init__(self, modelWeightsPath):
        self.model = models.load_model(modelWeightsPath, backbone_name='resnet50')
        self.model.predict_on_batch(np.zeros([1, 1, 1, 3]))  # warm up model
        self.session = K.get_session()
        self.graph = tf.get_default_graph()
        self.graph.finalize()  # finalize


    def predict_on_image(self, image, scoreThresh):
        image = preprocess_image(image)  # preprocess image for network
        image, scale = resize_image(image)

        with self.session.as_default():
            with self.graph.as_default():
                boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))

        boxes = np.divide(boxes, scale, out=boxes)  # boxes /= scale  # correct for image scale

        detections = [(box, label, score) for box, label, score in zip(boxes[0], labels[0], scores[0])
                      if score >= scoreThresh]
        boxes = (Box(d[0]) for d in detections)
        return boxes, detections

    def detect(self, frame, framePos, scoreThresh):
        return self.predict_on_image(frame, scoreThresh)


def test_case_1():
    """
    create and predict in different sequential threads
    """

    class TestThread(QThread):
        def __init__(self, detectorFactory, testFrame):
            super(TestThread, self).__init__()
            self.testFrame = testFrame
            self.detectorFactory = detectorFactory

        def run(self):
            detector = self.detectorFactory()
            boxes, detections = detector.detect(self.testFrame, 366, .85)
            print(len(list(boxes)))

    framePath = 'f_0366_24400.00_24.40.jpg'
    frame = cv2.imread(framePath)

    thread = TestThread(lambda: RetinanetPinDetector('../../modelWeights/retinanet_pins_inference.h5'), frame)
    thread.start()
    thread.wait()
    print('Thread 1 finished')
    thread = TestThread(lambda: RetinanetPinDetector('../../modelWeights/retinanet_pins_inference.h5'), frame)
    thread.start()
    thread.wait()


def test_case_2():
    """
    create in main thread and predict in different sequential threads
    """

    class TestThread(QThread):
        lock = Lock()

        def __init__(self, detector, testFrame, id):
            super(TestThread, self).__init__()
            self.id = id
            self.testFrame = testFrame
            self.detector = detector

        def run(self):
            for _ in range(10):
                boxes, detections = self.detector.detect(self.testFrame, 366, .85)
                with self.lock:
                    print('Thread ', int(self.currentThreadId()), self.id)

    framePath = 'f_0366_24400.00_24.40.jpg'
    frame = cv2.imread(framePath)

    detector = RetinanetPinDetector('../../modelWeights/retinanet_pins_inference.h5')

    thread1 = TestThread(detector, frame, 11111)
    thread2 = TestThread(detector, frame, 22222)

    thread1.start()
    thread2.start()

    thread1.wait()
    thread2.wait()


def main():
    app = QApplication([])
    test_case_2()


main()
