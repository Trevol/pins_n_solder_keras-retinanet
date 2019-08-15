from collections import deque

import cv2
import numpy as np

from detection.pins_tracking.v1.Colors import Colors
from detection.pins_tracking.v1.Constants import StabilizationLength
from detection.pins_tracking.v1.StatParams import StatParams
from utils import roundPoint, roundToInt


class Pin:
    def __init__(self, box, meanColor):
        self.box = box
        self._meanColors = deque([meanColor], maxlen=StabilizationLength)
        self.__colorStat = None
        self.withSolder = False

    @property
    def meanColors(self):
        return self._meanColors

    def __stabilized(self):
        return len(self._meanColors) == self._meanColors.maxlen

    @property
    def colorStat(self):
        assert self.__stabilized()
        return self.__colorStat

    def update(self, box, meanColor):
        self.box = box
        self._meanColors.append(meanColor)
        self.__updateStats()

    def __updateStats(self):
        if not self.__stabilized():
            return
        allColors = list(self._meanColors)
        colors = allColors[10:]
        mean = np.mean(colors, axis=0)
        std = np.std(colors, axis=0)
        median = np.median(allColors, axis=0)
        self.__colorStat = StatParams(mean, std, median)

    def draw(self, img):
        color = Colors.green if self.withSolder else Colors.yellow
        r = roundToInt(min(self.box.size[0], self.box.size[1]) / 4)  # min(w,h)/4
        cv2.circle(img, roundPoint(self.box.center), r, color, -1)
