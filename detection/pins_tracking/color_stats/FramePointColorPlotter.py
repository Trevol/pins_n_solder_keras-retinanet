from collections import deque

import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils.Timer import timeit


class FramePointColorPlotter:
    def __init__(self, dataQueueLen=300):
        self.point = None

        self.posData = deque(maxlen=dataQueueLen)
        self.colorData = deque(maxlen=300)

        fig, self.ax = plt.subplots()
        self.ax.set_ylim(0, 16777215)
        plt.show(block=False)

    def setPoint(self, point):
        self.point = point
        # clear data and plot
        self.posData.clear()
        self.colorData.clear()
        self.ax.clear()
        self.ax.set_ylim(0, 16777215)

    def drawPoint(self, img):
        if self.point is None:
            return
        x, y = self.point
        color = img[y, x]
        color = tuple(map(int, np.invert(color)))  # cant pass color as uint8 array...
        cv2.circle(img, self.point, 2, color=color, thickness=-1)

    @staticmethod
    def color24bit(img, point):
        x, y = point
        color = img[y, x]
        b = int(color[0])
        g = int(color[1])
        r = int(color[2])
        return b + (g << 8) + (r << 16)

    def plotColor(self, pos, img):
        if self.point is None:
            return
        self.posData.append(pos)
        color24 = self.color24bit(img, self.point)
        self.colorData.append(color24)

        with timeit():
            self.ax.clear()
            self.ax.set_ylim(0, 16777215)
            self.ax.scatter(self.posData, self.colorData, s=1)
            # plt.pause(.01)
            plt.draw()
