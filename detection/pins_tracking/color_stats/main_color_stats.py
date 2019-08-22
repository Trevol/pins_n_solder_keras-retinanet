import numpy as np
import cv2

from detection.pins_tracking.color_stats.FrameInfoPlotter import FrameInfoPlotter
from detection.pins_tracking.color_stats.RectSelection import RectSelection
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
    def color24bit(bgr):
        b = int(bgr[0])
        g = int(bgr[1])
        r = int(bgr[2])
        return b + (g << 8) + (r << 16)


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

        x1, y1 = self.rectSelection.pt1
        x2, y2 = self.rectSelection.pt2
        slice = self._frame[y1:y2 + 1, x1:x2 + 1]
        meanColor = np.mean(slice, axis=(0, 1))
        color24 = u.color24bit(meanColor)
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
