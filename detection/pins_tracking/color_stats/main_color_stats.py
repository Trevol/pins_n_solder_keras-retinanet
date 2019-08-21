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
    def __init__(self, frameSize, framesCount):
        super(PlottingVideoHandler, self).__init__(frameSize)
        self._frameScaleFactor = 1
        self.plotter = FramePointColorPlotter(framesCount)

    def frameReady(self, frame, framePos, framePosMsec, playback):
        self.plotter.plotColor(framePos, frame)
        self.plotter.drawPoint(frame)
        super(PlottingVideoHandler, self).frameReady(frame, framePos, framePosMsec, playback)

    def onMouse(self, evt, displayFrameX, displayFrameY, flags, param):
        if evt != cv2.EVENT_LBUTTONDOWN:
            return
        originalX = roundToInt(displayFrameX / self._frameScaleFactor)
        originalY = roundToInt(displayFrameY / self._frameScaleFactor)
        self.plotter.setPoint((originalX, originalY))
        if self._frame is not None:
            self.plotter.drawPoint(self._frame)
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
