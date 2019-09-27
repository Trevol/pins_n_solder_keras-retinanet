from collections import deque

from matplotlib import pyplot as plt


class FrameInfoPlotter:
    def __init__(self, lines, dataQueueLen=300, frameRangeToUpdate=10):
        self.framesPositions = deque(maxlen=dataQueueLen)
        self.linesData = [deque(maxlen=dataQueueLen) for l in lines]
        self.lines = lines
        self.setOfAxes = list(set(l.axes for l in self.lines))
        self.__eventConnections = self.__autoTightLayout(lines)
        self.__plotCounter = 0
        self.__frameRangeToUpdate = frameRangeToUpdate
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
        self.__plotCounter = 0
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

    def __timeToUpdate(self):
        return self.__plotCounter <= 100 or self.__plotCounter % self.__frameRangeToUpdate == 0

    def plot(self, framePos, values):
        if not self.__plot_shown:
            plt.show(block=False)
            self.__plot_shown = True

        self.framesPositions.append(framePos)
        for line, lineData, value in zip(self.lines, self.linesData, values):
            lineData.append(value)
            if self.__timeToUpdate():
                line.set_data(self.framesPositions, lineData)

        if self.__timeToUpdate():
            self.__setXlim(self.framesPositions[0], framePos)
            plt.draw()
        self.__plotCounter += 1
