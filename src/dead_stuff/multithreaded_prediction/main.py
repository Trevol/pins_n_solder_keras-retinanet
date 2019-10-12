from threading import Lock

import cv2
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication

import numpy as np

from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
from detection.Box import Box

from segmentation.MyVGGUnet import VGGUnet
from utils import remainderlessDividable

import tensorflow as tf
import keras.backend as K


class RetinanetPinDetector:
    def __init__(self, modelWeightsPath):
        self.model = models.load_model(modelWeightsPath, backbone_name='resnet50')
        self.model.predict_on_batch(np.zeros([1, 1, 1, 3]))  # warm up model

    def predict_on_image(self, image, scoreThresh):
        image = preprocess_image(image)  # preprocess image for network
        image, scale = resize_image(image)

        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))

        boxes = np.divide(boxes, scale, out=boxes)  # boxes /= scale  # correct for image scale

        detections = [(box, label, score) for box, label, score in zip(boxes[0], labels[0], scores[0])
                      if score >= scoreThresh]
        boxes = (Box(d[0]) for d in detections)
        return boxes, detections

    def detect(self, frame, framePos, scoreThresh):
        return self.predict_on_image(frame, scoreThresh)


class UnetSceneSegmentation:
    def __init__(self, weightsPath):
        self.input_height = remainderlessDividable(1080 // 2, 32, 1)
        self.input_width = remainderlessDividable(1920 // 2, 32, 1)
        self.n_classes = 6
        self.model = VGGUnet(self.n_classes, input_height=self.input_height, input_width=self.input_width)
        self.output_height = self.model.outputHeight
        self.output_width = self.model.outputWidth
        self.model.load_weights(weightsPath)
        self.model.predict_on_batch(np.zeros([1, 3, self.input_height, self.input_width]))  # warm up model

    def prepareBatch(self, image):
        image = cv2.resize(image, (self.input_width, self.input_height))
        image = image.astype(np.float32)
        image[:, :, 0] -= 103.939
        image[:, :, 1] -= 116.779
        image[:, :, 2] -= 123.68
        image = np.rollaxis(image, 2, 0)  # channel_first
        return np.expand_dims(image, 0)  # to batch [1, 3, h, w]

    def getSegmentationMap(self, frame, framePos):
        batch = self.prepareBatch(frame)
        predictions = self.model.predict(batch)[0]
        pixelClassProbabilities: np.ndarray = predictions.reshape(
            (self.output_height, self.output_width, self.n_classes))
        labelsImage = pixelClassProbabilities.argmax(axis=2)
        return labelsImage


class TestThread(QThread):
    lock = Lock()

    def __init__(self, session, graph, detector, segmentation, testFrame, id):
        super(TestThread, self).__init__()
        self.session = session
        self.graph = graph
        self.segmentation = segmentation
        self.id = id
        self.testFrame = testFrame
        self.detector = detector

    def run(self):
        try:
            with self.lock:
                print('Thread ', int(self.id), ' started!!!')
            with self.session.as_default():
                with self.graph.as_default():
                    for _ in range(10):
                        boxes, detections = self.detector.detect(self.testFrame, 366, .85)
                        self.segmentation.getSegmentationMap(self.testFrame, 366)
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

    session = K.get_session()
    graph = tf.get_default_graph()

    detector = RetinanetPinDetector('../../modelWeights/retinanet_pins_inference.h5')
    segmentation = UnetSceneSegmentation('../../modelWeights/unet_pins_25_0.000016_1.000000.hdf5')

    graph.finalize()

    thread1 = TestThread(session, graph, detector, segmentation, frame, 11111)
    thread2 = TestThread(session, graph, detector, segmentation, frame, 22222)
    thread3 = TestThread(session, graph, detector, segmentation, frame, 33333)
    thread4 = TestThread(session, graph, detector, segmentation, frame, 44444)
    thread5 = TestThread(session, graph, detector, segmentation, frame, 55555)

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

    session = K.get_session()
    graph = tf.get_default_graph()

    detector = RetinanetPinDetector('../../modelWeights/retinanet_pins_inference.h5')
    segmentation = UnetSceneSegmentation('../../modelWeights/unet_pins_25_0.000016_1.000000.hdf5')

    graph.finalize()

    thread1 = TestThread(session, graph, detector, segmentation, frame, 11111)
    thread2 = TestThread(session, graph, detector, segmentation, frame, 22222)
    thread3 = TestThread(session, graph, detector, segmentation, frame, 33333)
    thread4 = TestThread(session, graph, detector, segmentation, frame, 44444)
    thread5 = TestThread(session, graph, detector, segmentation, frame, 55555)

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
    # test_case_2()
    test_case_1()


main()
