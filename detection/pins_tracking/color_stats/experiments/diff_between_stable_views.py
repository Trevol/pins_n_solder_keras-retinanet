import cv2
import numpy as np

from utils.Timer import timeit
from utils.VideoPlayback import VideoPlayback
from utils.VideoPlaybackHandlerBase import VideoPlaybackHandlerBase
import utils.visualize


def playVideo():
    def files():
        yield '/HDD_DATA/Computer_Vision_Task/Video_6.mp4'
        # yield '/HDD_DATA/Computer_Vision_Task/Video_2.mp4'

    class VideoHandler(VideoPlaybackHandlerBase):
        def __init__(self, frameSize):
            super(VideoHandler, self).__init__(frameSize)
            self._frameScaleFactor = .85

        def processDisplayFrame(self, displayFrame0):
            return utils.visualize.putFramePos(displayFrame0, self._framePos, None)

    ################################################################
    for sourceVideoFile in files():
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=False)
        handler = VideoHandler(videoPlayback.frameSize())

        # framesRange = (4150, None)
        framesRange = None
        videoPlayback.play(range=framesRange, onFrameReady=handler.frameReady, onStateChange=handler.syncPlaybackState)
        cv2.waitKey()
        videoPlayback.release()
        handler.release()

    cv2.waitKey()


def getConsequentFrames():
    frames = (2436, 2633)
    # frames = (420, 586)
    p = VideoPlayback('/HDD_DATA/Computer_Vision_Task/Video_6.mp4', 1, autoplayInitially=False)
    f1 = p.readFrame(frames[0])
    f2 = p.readFrame(frames[1])
    return f1, f2


def makeDiff(f1, f2):
    grayF1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    grayF2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    # TODO: compute gradients and try extract peaks/ridges(хребты). By ridgeDetector?

    # наши "эллипсы" самае светлые - свет падает на них и хорошо отражается в камеру
    # cv2.subtract отриц. значения (более темные) устанавливает в 0, что и требуется
    grayDiff = cv2.subtract(grayF2, grayF1)

    thr, mask = cv2.threshold(grayDiff, 40, 255, cv2.THRESH_BINARY)
    return mask, grayDiff


def showDiff():
    def s(f):
        return f
        return f[600:, 350:1600]

    f1, f2 = getConsequentFrames()
    mask, grayDiff = makeDiff(f1, f2)
    cv2.imshow('grayDiff', s(grayDiff))
    cv2.imshow('mask', s(mask))
    while cv2.waitKey() != 27: pass


def main():
    # TODO: detect blobs on mask and grayDiff(with thresholding)
    showDiff()


main()
