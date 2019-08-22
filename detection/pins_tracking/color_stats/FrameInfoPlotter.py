from collections import deque

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from utils.Timer import timeit


class FrameInfoPlotter:
    def __init__(self, ylim, dataQueueLen=300):
        self.data = deque(maxlen=dataQueueLen)

        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(0, ylim)
        self.ax.set_xlim(0, 300)

        self.line = self.ax.add_line(Line2D([], [], markersize='1', marker='o', linestyle=''))
        self._eventId = self.fig.canvas.mpl_connect('resize_event', lambda e: plt.tight_layout())

        self.__plot_shown = False

    def __maximizeFigureWindow(self):
        self.fig.canvas.manager.window.showMaximized()

    def release(self):
        self.fig.canvas.mpl_disconnect(self._eventId)
        plt.close(self.fig)

    def clear(self):
        self.data.clear()
        self.line.set_data([], [])

    def __setXlim(self, minPos, maxPos):
        xMin, xMax = self.ax.get_xlim()
        if maxPos > xMax - 10:
            self.ax.set_xlim(minPos, maxPos + 100)

    def plot(self, framePos, value):
        if not self.__plot_shown:
            plt.show(block=False)
            self.__plot_shown = True

        self.data.append((framePos, value))

        # with timeit():
        dataAsArray = np.array(self.data)
        positions = dataAsArray[:, 0]
        values = dataAsArray[:, 1]
        self.line.set_data(positions, values)

        self.__setXlim(positions[0], positions[-1])
        plt.draw()
