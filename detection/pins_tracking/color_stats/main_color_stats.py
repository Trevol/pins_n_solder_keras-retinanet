import numpy as np
import cv2

from detection.pins_tracking.color_stats.FramePointColorPlotter import FramePointColorPlotter
from utils import roundToInt
from utils.VideoPlayback import VideoPlayback
from utils.VideoPlaybackHandlerBase import VideoPlaybackHandlerBase


def files():
    yield '/HDD_DATA/Computer_Vision_Task/Video_6.mp4'
    # yield '/HDD_DATA/Computer_Vision_Task/Video_2.mp4'


class PlottingVideoHandler(VideoPlaybackHandlerBase):
    max24bit = 16777215

    def __init__(self, frameSize, framesCount):
        super(PlottingVideoHandler, self).__init__(frameSize)
        self._frameScaleFactor = 1
        self.plotter = FramePointColorPlotter(self.max24bit, framesCount)
        self.framePoint = None
        self.displayFramePoint = None

    @staticmethod
    def color24bit(img, point):
        x, y = point
        color = img[y, x]
        b = int(color[0])
        g = int(color[1])
        r = int(color[2])
        return b + (g << 8) + (r << 16)

    @staticmethod
    def drawPoint(point, img):
        x, y = point
        color = img[y, x]
        color = tuple(map(int, np.invert(color)))  # cant pass color as uint8 array...
        cv2.circle(img, point, 2, color, thickness=-1)

    def getDisplayFrame(self):
        displayFrame = super(PlottingVideoHandler, self).getDisplayFrame()
        if self.framePoint:
            self.drawPoint(self.displayFramePoint, displayFrame)
        return displayFrame

    def __plotFrameValue(self):
        color24 = self.color24bit(self._frame, self.framePoint)
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
        self.drawPoint(self.displayFramePoint, self._displayFrame)
        self.showFrame()

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
    # save_frame_colors_at_point()
