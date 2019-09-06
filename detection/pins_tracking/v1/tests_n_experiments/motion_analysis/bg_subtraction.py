import cv2

from utils import resize
from utils.Timer import timeit
from utils.VideoPlayback import VideoPlayback
from utils.VideoPlaybackHandlerBase import VideoPlaybackHandlerBase


def files():
    yield '/HDD_DATA/Computer_Vision_Task/Video_6.mp4'
    # yield '/HDD_DATA/Computer_Vision_Task/Video_2.mp4'


class VideoHandler(VideoPlaybackHandlerBase):
    def __init__(self, frameSize):
        super(VideoHandler, self).__init__(frameSize)
        self._frameScaleFactor = 1
        self.bgSubtractor: cv2.BackgroundSubtractor = self.createBackgroudSubtractor()

    def createBackgroudSubtractor(self):
        return cv2.createBackgroundSubtractorKNN(history=500, detectShadows=False)
        # return cv2.bgsegm.createBackgroundSubtractorGSOC()

    def frameReady(self, frame, framePos, framePosMsec, playback):
        frame = resize(frame, .5)
        # frame = cv2.medianBlur(frame, 5)
        # frame = cv2.GaussianBlur(frame, (3, 3), 0)

        fgMask1 = self.bgSubtractor.apply(frame)
        bgImage1 = self.bgSubtractor.getBackgroundImage()

        cv2.imshow('fgMask1', resize(fgMask1, 0.5))
        cv2.imshow('bgImage1', resize(bgImage1, 0.5))

        super(VideoHandler, self).frameReady(frame, framePos, framePosMsec, playback)

    def release(self):
        self.bgSubtractor.clear()


def main():
    for sourceVideoFile in files():
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=False)
        handler = VideoHandler(videoPlayback.frameSize())
        endOfVideo = videoPlayback.playWithHandler(handler, range=(8000,))
        if endOfVideo:
            cv2.waitKey()
        videoPlayback.release()
        handler.release()


#########################
#########################
main()
