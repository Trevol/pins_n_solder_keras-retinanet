import cv2
import numpy as np

from utils.VideoPlayback import VideoPlayback
from utils.VideoPlaybackHandlerBase import VideoPlaybackHandlerBase
import utils.visualize


def files():
    yield '/HDD_DATA/Computer_Vision_Task/Video_6.mp4'
    # yield '/HDD_DATA/Computer_Vision_Task/Video_2.mp4'


class VideoHandler(VideoPlaybackHandlerBase):
    def processDisplayFrame(self, displayFrame0):
        return utils.visualize.putFramePos(displayFrame0, self._framePos, None)


def playVideo():
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


def makeDiff():
    p = VideoPlayback('/HDD_DATA/Computer_Vision_Task/Video_6.mp4', 1, autoplayInitially=False)
    f1 = p.readFrame(420)
    f2 = p.readFrame(586)
    bgrDiff = cv2.absdiff(f1, f2)

    total = np.int32(bgrDiff[..., 0]) + bgrDiff[..., 1] + bgrDiff[..., 2]
    npGrayDiff = np.divide(total, 3).astype(np.uint8)
    cvGrayDiff = cv2.cvtColor(bgrDiff, cv2.COLOR_BGR2GRAY)

    # TODO: f1 and f2 to gray
    grayF1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    grayF2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    # TODO: compute gradients and try extract peaks/ridges(хребты). By ridgeDetector?
    # cv2.imshow('bgrDiff', s(bgrDiff))
    # cv2.imshow('npGrayDiff', s(npGrayDiff))
    # cv2.imshow('cvGrayDiff', s(cvGrayDiff))
    # cv2.imshow('diffOfDiffs', s(diffOfDiffs))
    # cv2.imshow('f1', s(f1))
    # cv2.imshow('f2', s(f2))
    # cv2.imshow('grayF1', s(grayF1))
    # cv2.imshow('grayF2', s(grayF2))

    # наши "эллипсы" самае светлые - свет падает на них и хорошо отражается в камеру
    gg = cv2.subtract(grayF2, grayF1)  # cv2.subtract отриц. значения (более темные) устанавливает в 0, что и требуется

    cv2.imshow('gg', gg)
    while cv2.waitKey() != 27: pass


def s(f):
    return f[300:, 350:1600]


makeDiff()
