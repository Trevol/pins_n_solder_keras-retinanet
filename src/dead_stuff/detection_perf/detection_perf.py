import cv2
from detection.RetinanetPinDetector import RetinanetPinDetector
from models.weights.config import retinanet_pins_weights
from utils.Timer import timeit
import numpy as np


def main():
    frame = cv2.imread('../images/f_0366_24400.00_24.40.jpg')
    detector = RetinanetPinDetector(retinanet_pins_weights, True)

    for _ in range(7):
        with timeit():
            boxes, _ = detector.detect(frame, 0, .85)


def main():
    imagenetMean = [103.939, 116.779, 123.68]

    img = np.zeros([3, 1080, 1920], np.float32)
    for _ in range(5):
        with timeit():
            img[0] -= imagenetMean[0]
            img[1] -= imagenetMean[1]
            img[2] -= imagenetMean[2]
    print('--------------------')
    img = np.zeros([1080, 1920, 3], np.float32)
    for _ in range(5):
        with timeit():
            img[..., 0] -= imagenetMean[0]
            img[..., 1] -= imagenetMean[1]
            img[..., 2] -= imagenetMean[2]


main()
