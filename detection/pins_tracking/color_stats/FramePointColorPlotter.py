from collections import deque

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from utils.Timer import timeit


class FramePointColorPlotter:
    def __init__(self, dataQueueLen=300):
        self.point = None

        # self.posData = deque(maxlen=dataQueueLen)
        # self.colorData = deque(maxlen=300)
        self.data = deque(maxlen=dataQueueLen)

        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(0, 16777215)
        # self.line = Line2D([1, 2], [1000000, 200000])
        # self.ax.add_line(self.line)
        self.line, = self.ax.plot([1, 2], [1000000, 200000], 'go-', linestyle='', markersize=1)
        plt.show(block=False)

    def setPoint(self, point):
        self.point = point
        self.data.clear()
        self.line.set_data([], [])
        # self.ax.set_ylim(0, 16777215)

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
        return b + (g << 8) + (r << 16), b, g, r

    def plotColor(self, pos, img):
        if self.point is None:
            return
        color24, b, g, r = self.color24bit(img, self.point)
        self.data.append((pos, color24, b, g, r))

        with timeit():
            dataAsArray = np.array(self.data)
            # self.ax.clear()
            # self.ax.set_ylim(0, 16777215)
            # todo: plot with self.ax.line
            # l, = self.ax.plot(dataAsArray[:, 0], dataAsArray[:, 1], 'go-', linestyle='', markersize=1)
            # plt.pause(.01)
            self.line.set_data(dataAsArray[:, 0], dataAsArray[:, 1])
            self.line.set_linestyle('')
            self.line.set_color('g')
            self.line.set_markersize(1)
            # self.ax.set_ylim(0, 16777215)
            # self.ax.draw(None)
            # return
            # plt.draw()
            # self.ax.figure.canvas.draw()
            self.ax.draw_artist(self.line)
            self.fig.canvas.draw_idle()