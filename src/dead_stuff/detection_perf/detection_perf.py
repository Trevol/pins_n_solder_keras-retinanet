import cv2
from detection.RetinanetPinDetector import RetinanetPinDetector
from models.weights.config import retinanet_pins_weights
from utils.Timer import timeit


def main():
    frame = cv2.imread('../images/f_0366_24400.00_24.40.jpg')
    detector = RetinanetPinDetector(retinanet_pins_weights, True)

    for _ in range(5):
        with timeit():
            boxes, _ = detector.detect(frame, 0, .85)


main()
