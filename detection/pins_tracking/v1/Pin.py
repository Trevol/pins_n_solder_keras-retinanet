from collections import deque

import cv2
import numpy as np

from detection.pins_tracking.v1.Colors import Colors
from detection.pins_tracking.v1.Constants import StabilizationLength
from detection.pins_tracking.v1.StatParams import StatParams
from utils import roundPoint, roundToInt


class Pin:
    def __init__(self, box):
        self.box = box
        self.withSolder = False

    def update(self, box):
        self.box = box

    def draw(self, img):
        color = Colors.green if self.withSolder else Colors.yellow
        r = roundToInt(min(self.box.size[0], self.box.size[1]) / 4)  # min(w,h)/4
        cv2.circle(img, roundPoint(self.box.center), r, color, -1)
