import numpy as np
import cv2

from detection.pins_tracking.color_stats.FrameInfoPlotter import FrameInfoPlotter
from detection.pins_tracking.color_stats.RectSelection import RectSelection
from utils import colorChannelsTo24bit
from utils.VideoPlayback import VideoPlayback
from utils.VideoPlaybackHandlerBase import VideoPlaybackHandlerBase


def files():
    yield '/HDD_DATA/Computer_Vision_Task/Video_6.mp4'
    # yield '/HDD_DATA/Computer_Vision_Task/Video_2.mp4'


class ColorExtraction:
    @staticmethod
    def areaMeanColor24(img, rectSelection):
        x1, y1 = rectSelection.pt1
        x2, y2 = rectSelection.pt2
        area = img[y1:y2 + 1, x1:x2 + 1]
        meanColor = np.mean(area, axis=(0, 1))
        return colorChannelsTo24bit(meanColor)

    @staticmethod
    def cornersMeanColor24(img, rectSelection):
        b24 = colorChannelsTo24bit
        x1, y1 = rectSelection.pt1
        x2, y2 = rectSelection.pt2
        x3, y3 = x1, y2
        x4, y4 = x2, y1
        return (b24(img[y1, x1]) + b24(img[y2, x2]) + b24(img[y3, x3]) + b24(img[y4, x4])) // 4


class PlottingVideoHandler(VideoPlaybackHandlerBase):
    max24bit = 16777215

    def __init__(self, frameSize, framesCount):
        super(PlottingVideoHandler, self).__init__(frameSize)
        # self._frameScaleFactor = 1
        self.plotter = FrameInfoPlotter(self.max24bit, framesCount)
        self.rectSelection = RectSelection(self._frameScaleFactor)

    def processDisplayFrame(self, displayFrame0):
        if self.rectSelection.selected():
            return self.rectSelection.draw(displayFrame0.copy())
        return super(PlottingVideoHandler, self).processDisplayFrame(displayFrame0)

    def __plotFrameValue(self):
        if not self.rectSelection.selected():
            return
        # color24 = ColorExtraction.areaMeanColor24(self._frame, self.rectSelection)
        color24 = ColorExtraction.cornersMeanColor24(self._frame, self.rectSelection)
        self.plotter.plot(self._framePos, color24)
        pass

    def frameReady(self, frame, framePos, framePosMsec, playback):
        super(PlottingVideoHandler, self).frameReady(frame, framePos, framePosMsec, playback)
        self.__plotFrameValue()

    def onMouse(self, evt, displayFrameX, displayFrameY, flags, param):
        if evt != cv2.EVENT_LBUTTONDOWN or self._frame is None:
            return
        self.rectSelection.addDisplayPoint(displayFrameX, displayFrameY)

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
