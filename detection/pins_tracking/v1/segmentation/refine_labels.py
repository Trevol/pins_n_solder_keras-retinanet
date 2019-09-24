import glob
from collections import deque
import cv2
import numpy as np

from detection.pins_tracking.v1.segmentation.classesMeta import BGR
from detection.pins_tracking.v1.segmentation.pin_utils import colorizeLabel


def refine():
    labelsDir = '/HDD_DATA/Computer_Vision_Task/frames_6/not_augmented_base_vgg16_more_images_25/'
    labelsPaths = sorted(glob.glob(labelsDir + "*_label.png"))
    q = deque(maxlen=5)
    for path in labelsPaths:
        labelImg = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        q.append(labelImg)

        refined = np.mean(q, axis=0)
        refined = np.uint8(np.round(refined, 0))

        cv2.imshow('refined label', colorizeLabel(refined, BGR))
        cv2.imshow('label', colorizeLabel(labelImg, BGR))
        if cv2.waitKey() == 27:
            break


def main():
    refine()


main()
