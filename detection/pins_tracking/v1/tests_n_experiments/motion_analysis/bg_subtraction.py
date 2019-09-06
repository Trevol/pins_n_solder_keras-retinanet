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
        knn = cv2.createBackgroundSubtractorKNN(history=500, detectShadows=True)
        knn.setShadowValue(0)
        return knn
        # return cv2.bgsegm.createBackgroundSubtractorGSOC()
        # return cv2.bgsegm.createBackgroundSubtractorLSBP()
        # return cv2.bgsegm.createBackgroundSubtractorMOG()

        mog2 = cv2.createBackgroundSubtractorMOG2(history=800, detectShadows=False)
        # mog2.setShadowValue(0)
        return mog2

    def frameReady(self, frame, framePos, framePosMsec, playback):
        frame = resize(frame, .5)
        # frame = cv2.medianBlur(frame, 5)
        # frame = cv2.GaussianBlur(frame, (3, 3), 0)

        with timeit():
            fgMask1 = self.bgSubtractor.apply(frame)

        with timeit(autoreport=False) as t:
            cntNonZero = cv2.countNonZero(fgMask1)
            print('cntNonZero', t.getDuration(), cntNonZero)

        with timeit(autoreport=False) as t:
            retval, labels = cv2.connectedComponents(fgMask1)
            print('connectedComponents', t.getDuration(), retval, labels.shape)

        with timeit(autoreport=False) as t:
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(fgMask1)
            print('connectedComponentsWithStat', t.getDuration(), retval, labels.shape, stats.shape, centroids.shape)

        print('-------------------------------')
        # bgImage1 = self.bgSubtractor.getBackgroundImage()

        niters = 3
        refined = cv2.erode(fgMask1, None, None, None, 1)
        refined = cv2.dilate(refined, None, refined, None, niters)
        refined = cv2.erode(refined, None, refined, None, niters * 2)
        refined = cv2.dilate(refined, None, refined, None, niters)

        cv2.imshow('fgMask1', resize(fgMask1, 1))
        cv2.imshow('refined', resize(refined, 1))
        # cv2.imshow('bgImage1', resize(bgImage1, 1.5))

        super(VideoHandler, self).frameReady(frame, framePos, framePosMsec, playback)

    def release(self):
        self.bgSubtractor.clear()


def main():
    for sourceVideoFile in files():
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=False)
        handler = VideoHandler(videoPlayback.frameSize())
        endOfVideo = videoPlayback.playWithHandler(handler)
        if endOfVideo:
            cv2.waitKey()
        videoPlayback.release()
        handler.release()


#########################
#########################
main()
