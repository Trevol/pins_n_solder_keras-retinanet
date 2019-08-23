from collections import deque

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from utils.Timer import timeit


class FrameInfoPlotter:
    def __init__(self, lines, dataQueueLen=300):
        self.framesPositions = deque(maxlen=dataQueueLen)
        self.linesData = [deque(maxlen=dataQueueLen) for l in lines]
        self.lines = lines
        self.setOfAxes = list(set(l.axes for l in self.lines))
        self.__eventConnections = self.__autoTightLayout(lines)
        self.__plot_shown = False

    @staticmethod
    def __autoTightLayout(lines):
        eventConnections = []
        canvasSet = {l.figure.canvas for l in lines}
        for canvas in canvasSet:
            eventId = canvas.mpl_connect('resize_event', lambda e: e.canvas.figure.tight_layout())
            eventConnections.append((eventId, canvas))
        return eventConnections

    def __maximizeFigureWindow(self):
        figures = {l.figure.canvas for l in self.lines}
        for figure in figures:
            figure.canvas.manager.window.showMaximized()

    def release(self):
        self.clear()
        for eventId, canvas in self.__eventConnections:
            canvas.mpl_disconnect(eventId)
            plt.close(canvas.figure)

    def clear(self):
        self.framesPositions.clear()
        for data, line in zip(self.linesData, self.lines):
            line.set_data([], [])
            data.clear()

    def __setXlim(self, minPos, maxPos):
        xMin, xMax = self.setOfAxes[0].get_xlim()
        if maxPos < xMax - 10:  # no need to reset limit
            return
        for ax in self.setOfAxes:
            ax.set_xlim(minPos, maxPos + 100)

    def plot(self, framePos, values):
        if not self.__plot_shown:
            plt.show(block=False)
            self.__plot_shown = True

        self.framesPositions.append(framePos)
        for line, lineData, value in zip(self.lines, self.linesData, values):
            lineData.append(value)
            line.set_data(self.framesPositions, lineData)

        self.__setXlim(self.framesPositions[0], framePos)
        plt.draw()

    def plot___OLD(self, framePos, values):
        if not self.__plot_shown:
            plt.show(block=False)
            self.__plot_shown = True

        self.data.append((framePos, values))

        # with timeit():
        dataAsArray = np.array(self.data)
        positions = dataAsArray[:, 0]
        values = dataAsArray[:, 1]
        self.line.set_data(positions, values)

        self.__setXlim(positions[0], positions[-1])
        plt.draw()

# class FrameInfoPlotter:
#     def __init__(self, ylim, dataQueueLen=300):
#         self.data = deque(maxlen=dataQueueLen)
#
#         self.fig, self.ax = plt.subplots()
#         self.ax.set_ylim(0, ylim)
#         self.ax.set_xlim(0, 300)
#
#         self.line = self.ax.add_line(Line2D([], [], markersize='1', marker='o', linestyle=''))
#         self._eventId = self.fig.canvas.mpl_connect('resize_event', lambda e: plt.tight_layout())
#
#         self.__plot_shown = False
#
#     def __maximizeFigureWindow(self):
#         self.fig.canvas.manager.window.showMaximized()
#
#     def release(self):
#         self.clear()
#         self.fig.canvas.mpl_disconnect(self._eventId)
#         plt.close(self.fig)
#
#     def clear(self):
#         self.data.clear()
#         self.line.set_data([], [])
#
#     def __setXlim(self, minPos, maxPos):
#         xMin, xMax = self.ax.get_xlim()
#         if maxPos > xMax - 10:
#             self.ax.set_xlim(minPos, maxPos + 100)
#
#     def plot(self, framePos, value):
#         if not self.__plot_shown:
#             plt.show(block=False)
#             self.__plot_shown = True
#
#         self.data.append((framePos, value))
#
#         # with timeit():
#         dataAsArray = np.array(self.data)
#         positions = dataAsArray[:, 0]
#         values = dataAsArray[:, 1]
#         self.line.set_data(positions, values)
#
#         self.__setXlim(positions[0], positions[-1])
#         plt.draw()
