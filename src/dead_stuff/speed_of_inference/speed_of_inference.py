from cv2.bioinspired import RETINA_COLOR_RANDOM

from detection.PinDetector import PinDetector
from models.ModelsContext import ModelsContext
import numpy as np
import cv2

from models.weights.config import retinanet_pins_weights
from utils import resize
from utils.Timer import timeit

import numpy as np
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image

from detection.Box import Box
from detection.PinDetector import PinDetector


class RetinanetPinDetector(PinDetector):
    def __init__(self, modelWeightsPath, warmup=True):
        self.model = models.load_model(modelWeightsPath, backbone_name='resnet50')
        if warmup:
            self._warmupModel()

    def _warmupModel(self):
        warmupImg = np.zeros([1, 1, 1, 3])
        self.model.predict_on_batch(warmupImg)

    @staticmethod
    def predict_on_image(model, image, scoreThresh):
        image = preprocess_image(image)  # preprocess image for network
        image, scale = resize_image(image, min_side=432, max_side=768)

        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

        boxes = np.divide(boxes, scale, out=boxes)  # boxes /= scale  # correct for image scale

        detections = [(box, label, score) for box, label, score in zip(boxes[0], labels[0], scores[0])
                      if score >= scoreThresh]
        boxes = (Box(d[0]) for d in detections)
        return boxes, detections

    def detect(self, frame, framePos, scoreThresh):
        return self.predict_on_image(self.model, frame, scoreThresh)


def main():
    detector = RetinanetPinDetector(retinanet_pins_weights)

    img_1920_1080_real = cv2.imread('f_0366_24400.00_24.40.jpg')
    img_1920_1080_zeros = np.zeros([1080, 1920, 3], np.uint8)

    img_960_540_zeros = np.zeros([540, 960, 3], np.uint8)
    img_960_540_real = resize(img_1920_1080_real, .4)

    for _ in range(5):
        with timeit():
            b, _ = detector.detect(img_1920_1080_real, 0, .85)
        print('bbb', len(list(b)))
        # with timeit():
        #     detector.detect(img_1920_1080_zeros, 0, .85)
        with timeit():
            b, _ = detector.detect(img_960_540_real, 0, .85)
        print('bbb', len(list(b)))
        # with timeit():
        #     detector.detect(img_960_540_zeros, 0, .85)

        print('-------------------------')


main()
