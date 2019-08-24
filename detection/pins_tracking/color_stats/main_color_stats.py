import numpy as np
import cv2
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from detection.pins_tracking.color_stats.ColorExtraction import ColorExtraction
from detection.pins_tracking.color_stats.FrameInfoPlotter import FrameInfoPlotter
from detection.pins_tracking.color_stats.RectSelection import RectSelection
from utils.VideoPlayback import VideoPlayback
from utils.VideoPlaybackHandlerBase import VideoPlaybackHandlerBase
import utils.visualize


def files():
    yield '/HDD_DATA/Computer_Vision_Task/Video_6.mp4'
    # yield '/HDD_DATA/Computer_Vision_Task/Video_2.mp4'


class PlottingVideoHandler(VideoPlaybackHandlerBase):
    max24bit = 16777215

    @staticmethod
    def configureLines():
        fig = plt.figure()
        ax = fig.subplots()
        ax.set_ylim(0, 255)
        ax.set_xlim(0, 300)  # initial limit

        bLine = ax.add_line(Line2D([], [], markersize='1', marker='o', linestyle='', color='b'))
        gLine = ax.add_line(Line2D([], [], markersize='1', marker='o', linestyle='', color='g'))
        rLine = ax.add_line(Line2D([], [], markersize='1', marker='o', linestyle='', color='r'))
        return bLine, gLine, rLine

    def __init__(self, frameSize, framesCount):
        super(PlottingVideoHandler, self).__init__(frameSize)
        self._frameScaleFactor = 1
        self.plotter = FrameInfoPlotter(self.configureLines(), framesCount)
        self.rectSelection = RectSelection(self._frameScaleFactor)

    def processDisplayFrame(self, displayFrame0):
        utils.visualize.putFramePos(displayFrame0, self._framePos, None)
        if self.rectSelection.selected():
            return self.rectSelection.draw(displayFrame0.copy())
        return super(PlottingVideoHandler, self).processDisplayFrame(displayFrame0)

    def __plotFrameValue(self):
        if not self.rectSelection.selected():
            return
        meanBgr = ColorExtraction.areaMeanColor(self._frame, self.rectSelection)
        self.plotter.plot(self._framePos, meanBgr)

    def frameReady(self, frame, framePos, framePosMsec, playback):
        super(PlottingVideoHandler, self).frameReady(frame, framePos, framePosMsec, playback)
        self.__plotFrameValue()

    def onMouse(self, evt, displayX, displayY, flags, param):
        if self._frame is None:
            return
        stateChanged = self.rectSelection.mouseEvent(evt, displayX, displayY, flags, param)
        if stateChanged:
            self.plotter.clear()
            self.__plotFrameValue()
            self.refreshDisplayFrame()

    def release(self):
        super(PlottingVideoHandler, self).release()
        self.plotter.release()


def main():
    for sourceVideoFile in files():
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=False)
        handler = PlottingVideoHandler(videoPlayback.frameSize(), videoPlayback.framesCount())

        # framesRange = (4150, None)
        framesRange = None
        videoPlayback.play(range=framesRange, onFrameReady=handler.frameReady, onStateChange=handler.syncPlaybackState)
        videoPlayback.release()
        cv2.waitKey()
        handler.release()

    cv2.waitKey()


np.seterr(all='raise')
if __name__ == '__main__':
    main()
