import os
import time

import numpy as np
import cv2
import resource

import psutil

from detection.csv_cache.DetectionsCSV import DetectionsCSV
import utils.visualize
from detection.pins_tracking.v1.PinDetector import PinDetector, PickledDictionaryPinDetector
from utils import resize
from utils.VideoPlayback import VideoPlayback
from utils import videoWriter

from detection.pins_tracking.v1.TechProcessTracker import TechProcessTracker
from utils.VideoPlaybackHandlerBase import VideoPlaybackHandlerBase


class TechProcessVideoHandler(VideoPlaybackHandlerBase):
    def __init__(self, frameSize, pinDetector: PinDetector):
        super(TechProcessVideoHandler, self).__init__(frameSize)
        self.techProcessTracker = TechProcessTracker(pinDetector)

    def frameReady(self, frame, framePos, framePosMsec, playback):
        self.techProcessTracker.track(framePos, framePosMsec, frame)
        super(TechProcessVideoHandler, self).frameReady(frame, framePos, framePosMsec, playback)

    def processDisplayFrame(self, displayFrame0):
        utils.visualize.putFramePos((10, 40), displayFrame0, self._framePos, self._framePosMsec)
        self.techProcessTracker.drawStats((10, 110), displayFrame0)
        self.techProcessTracker.draw(displayFrame0)
        return displayFrame0

    def release(self):
        super(TechProcessVideoHandler, self).release()


def files():
    yield ('/HDD_DATA/Computer_Vision_Task/Video_6.mp4',
           '/HDD_DATA/Computer_Vision_Task/Video_6_result.mp4',
           '../../csv_cache/data/detections_video6.pcl')

    # yield ('/HDD_DATA/Computer_Vision_Task/Video_2.mp4',
    #        '/HDD_DATA/Computer_Vision_Task/Video_2_result.mp4',
    #        '../../csv_cache/data/detections_video2.pcl')


def printMemoryUsage():
    print(psutil.Process().memory_info())  # in bytes


def main():
    printMemoryUsage()

    def getFramesRange():
        # framesRange = (4150, None)
        # framesRange = (8100, None)
        framesRange = None
        return framesRange

    np.seterr(all='raise')
    for sourceVideoFile, resultVideo, pclFile in files():
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=False)
        handler = TechProcessVideoHandler(videoPlayback.frameSize(), PickledDictionaryPinDetector(pclFile))

        endOfVideo = videoPlayback.playWithHandler(handler, range=getFramesRange())
        printMemoryUsage()

        if endOfVideo:
            cv2.waitKey()

        videoPlayback.release()
        handler.release()

    cv2.waitKey()


if __name__ == '__main__':
    main()
