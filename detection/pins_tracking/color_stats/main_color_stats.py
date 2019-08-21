import numpy as np
import cv2

from detection.pins_tracking.color_stats.FrameInfoPlotter import FrameInfoPlotter
from utils import roundToInt
from utils.VideoPlayback import VideoPlayback
from utils.VideoPlaybackHandlerBase import VideoPlaybackHandlerBase


def files():
    yield '/HDD_DATA/Computer_Vision_Task/Video_6.mp4'
    # yield '/HDD_DATA/Computer_Vision_Task/Video_2.mp4'


class u:  # u - utils
    @staticmethod
    def drawPoint(point, img):
        x, y = point
        color = img[y, x]
        color = tuple(map(int, np.invert(color)))  # cant pass color as uint8 array...
        return cv2.circle(img, point, 2, color, thickness=-1)

    @staticmethod
    def color24bit(img, point):
        x, y = point
        color = img[y, x]
        b = int(color[0])
        g = int(color[1])
        r = int(color[2])
        return b + (g << 8) + (r << 16)


class RectSelection:
    def __init__(self, displayScale):
        self.displayScale = displayScale
        self.pt1 = self.pt2 = None

    def hasDrawing(self):
        return self.pt1 is not None or self.pt2 is not None

    def displayPt(self, pt):
        x = roundToInt(pt[0] * self.displayScale)
        y = roundToInt(pt[1] * self.displayScale)
        return (x, y)

    def draw(self, img):
        if not self.hasDrawing():
            return img
        color = (0, 0, 200)
        thickness = 1
        if self.pt1 == self.pt2:
            thickness = 3
        return cv2.rectangle(img, self.displayPt(self.pt1), self.displayPt(self.pt2), color, thickness)

    def addDisplayPoint(self, displayX, displayY):
        originalX = roundToInt(displayX / self.displayScale)
        originalY = roundToInt(displayY / self.displayScale)
        pt = (originalX, originalY)
        if self.pt1 is None and self.pt2 is None:
            self.pt1 = self.pt2 = pt
        elif self.pt1 is not None and self.pt2 != self.pt1:
            self.pt1 = self.pt2 = None
        elif self.pt1 is not None and self.pt2 == self.pt1:
            self.pt2 = pt


class PlottingVideoHandler(VideoPlaybackHandlerBase):
    max24bit = 16777215

    def __init__(self, frameSize, framesCount):
        super(PlottingVideoHandler, self).__init__(frameSize)
        # self._frameScaleFactor = 1
        self.plotter = FrameInfoPlotter(self.max24bit, framesCount)
        self.rectSelection = RectSelection(self._frameScaleFactor)

    def processDisplayFrame(self, displayFrame0):
        if self.rectSelection.hasDrawing():
            return self.rectSelection.draw(displayFrame0.copy())
        return super(PlottingVideoHandler, self).processDisplayFrame(displayFrame0)

    def __plotFrameValue(self):
        # if self.framePoint:
        #     color24 = u.color24bit(self._frame, self.framePoint)
        #     self.plotter.plot(self._framePos, color24)
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


class PlottingVideoHandler_OLD(VideoPlaybackHandlerBase):
    max24bit = 16777215

    def __init__(self, frameSize, framesCount):
        super(PlottingVideoHandler, self).__init__(frameSize)
        # self._frameScaleFactor = 1
        self.plotter = FrameInfoPlotter(self.max24bit, framesCount)
        self.framePoint = None
        self.displayFramePoint = None

    def processDisplayFrame(self, displayFrame0):
        if self.framePoint:
            return u.drawPoint(self.displayFramePoint, displayFrame0.copy())
        return super(PlottingVideoHandler, self).processDisplayFrame(displayFrame0)

    def __plotFrameValue(self):
        color24 = u.color24bit(self._frame, self.framePoint)
        self.plotter.plot(self._framePos, color24)

    def frameReady(self, frame, framePos, framePosMsec, playback):
        super(PlottingVideoHandler, self).frameReady(frame, framePos, framePosMsec, playback)
        if self.framePoint:
            self.__plotFrameValue()

    def onMouse(self, evt, displayFrameX, displayFrameY, flags, param):
        if self._frame is None:
            return
        if evt != cv2.EVENT_LBUTTONDOWN:
            return
        originalX = roundToInt(displayFrameX / self._frameScaleFactor)
        originalY = roundToInt(displayFrameY / self._frameScaleFactor)
        self.framePoint = (originalX, originalY)
        self.displayFramePoint = (displayFrameX, displayFrameY)
        self.plotter.clear()
        self.__plotFrameValue()
        self.refreshDisplayFrame()

    def release(self):
        super(PlottingVideoHandler, self).release()
        self.plotter.release()
