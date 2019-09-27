import math

import numpy as np
import cv2
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib import gridspec

from dead_stuff.color_stats.ColorExtraction import ColorExtraction
from dead_stuff.color_stats.FrameInfoPlotter import FrameInfoPlotter
from dead_stuff.color_stats.MultiPointSelection import MultiPointSelection
from utils import VideoPlayback
from utils.VideoPlaybackHandlerBase import VideoPlaybackHandlerBase
import utils


def files():
    yield '/HDD_DATA/Computer_Vision_Task/Video_6.mp4'
    # yield '/HDD_DATA/Computer_Vision_Task/Video_2.mp4'


class PlottingVideoHandler(VideoPlaybackHandlerBase):
    max24bit = 16777215

    @staticmethod
    def configureLines(selectedPoints):
        assert any(selectedPoints)

        nCols = 2
        nRows = math.ceil(len(selectedPoints) / nCols)

        grid = gridspec.GridSpec(nRows, nCols)
        fig = plt.figure()
        lines = []
        for i in range(len(selectedPoints)):
            nRow = i // nCols
            nCol = i % nCols
            subplotSpec = grid[nRow, nCol]
            ax = fig.add_subplot(subplotSpec)
            ax.set_ylim(0, 255)
            ax.set_xlim(0, 300)  # initial limit
            hLine = ax.add_line(Line2D([], [], markersize='1', marker='o', linestyle='', color='#808000'))  # olive
            sLine = ax.add_line(Line2D([], [], markersize='1', marker='o', linestyle='', color='#808080'))  # gray
            vLine = ax.add_line(Line2D([], [], markersize='1', marker='o', linestyle='', color='#800080'))  # Purple
            lines.extend([hLine, sLine, vLine])

        return lines

    def __init__(self, frameSize, framesCount):
        super(PlottingVideoHandler, self).__init__(frameSize)
        self._frameScaleFactor = 1
        self.framesCount = framesCount
        self.plotter = None
        # self.plotter = FrameInfoPlotter(self.configureLines(), framesCount)
        # self.selection = RectSelection(self._frameScaleFactor)
        self.selection = MultiPointSelection(self._frameScaleFactor)

    def processDisplayFrame(self, displayFrame0):
        utils.visualize.putFramePos(displayFrame0, self._framePos, None)
        if self.selection.selected():
            return self.selection.draw(displayFrame0.copy())
        return super(PlottingVideoHandler, self).processDisplayFrame(displayFrame0)

    def __plotFrameValue(self):
        if not self.selection.selected():
            return
        hsvColors = ColorExtraction.multiPointSelectionHsvColors(self._frame, self.selection)
        hsvChannels = [ch for hsv in hsvColors for ch in hsv ]
        self.plotter.plot(self._framePos, hsvChannels)

    def frameReady(self, frame, framePos, framePosMsec, playback):
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        super(PlottingVideoHandler, self).frameReady(frame, framePos, framePosMsec, playback)
        self.__plotFrameValue()

    def onMouse(self, evt, displayX, displayY, flags, param):
        if self._frame is None:
            return
        stateChanged = self.selection.mouseEvent(evt, displayX, displayY, flags, param)
        if stateChanged:
            if self.plotter:
                self.plotter.release()
            if self.selection.selected():
                self.plotter = FrameInfoPlotter(self.configureLines(self.selection.points), self.framesCount)
                self.__plotFrameValue()
            self.refreshDisplayFrame()

    def release(self):
        super(PlottingVideoHandler, self).release()
        if self.plotter:
            self.plotter.release()


def main():
    for sourceVideoFile in files():
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=False)
        handler = PlottingVideoHandler(videoPlayback.frameSize(), videoPlayback.framesCount())

        # framesRange = (4150, None)
        framesRange = None
        videoPlayback.playWithHandler(handler, framesRange)
        videoPlayback.release()
        cv2.waitKey()
        handler.release()

    cv2.waitKey()


np.seterr(all='raise')
if __name__ == '__main__':
    main()
