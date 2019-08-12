import numpy as np
import cv2
from detection.csv_cache.DetectionsCSV import DetectionsCSV
import utils.visualize
from utils import resize
from utils.VideoPlayback import VideoPlayback
from utils import videoWriter

from detection.pins_tracking.v1.TechProcessTracker import TechProcessTracker
from detection.pins_tracking.v1.VideoConfig import video6SolderConfig


class VideoHandler:
    winname = 'Video'

    def __init__(self, frameSize):
        cv2.namedWindow(self.winname)
        cv2.setMouseCallback(self.winname, self.onMouse)
        self.__frameScaleFactor = 0.7 if frameSize[0] >= 1900 else 1

    def release(self):
        cv2.destroyWindow(self.winname)

    def onMouse(self, evt, x, y, flags, param):
        if evt != cv2.EVENT_LBUTTONUP:
            return

    def syncPlaybackState(self, frameDelay, autoPlay, framePos, framePosMsec, playback):
        autoplayLabel = 'ON' if autoPlay else 'OFF'
        stateTitle = f'{self.winname} (FrameDelay: {frameDelay}, Autoplay: {autoplayLabel})'
        cv2.setWindowTitle(self.winname, stateTitle)

    def frameReady(self, frame, framePos, framePosMsec, playback):
        imshowFrame = resize(frame, self.__frameScaleFactor)
        cv2.imshow(self.winname, imshowFrame)


def files():
    yield '/HDD_DATA/Computer_Vision_Task/Video_6.mp4'

    # yield '/HDD_DATA/Computer_Vision_Task/Video_2.mp4'


def main():
    for sourceVideoFile in files():
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=False)
        handler = VideoHandler(videoPlayback.frameSize())

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
