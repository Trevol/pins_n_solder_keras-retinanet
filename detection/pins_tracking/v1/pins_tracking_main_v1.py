import os
import time

import numpy as np
import cv2
import resource

import psutil

from detection.csv_cache.DetectionsCSV import DetectionsCSV
import utils.visualize
from utils import resize
from utils.VideoPlayback import VideoPlayback
from utils import videoWriter

from detection.pins_tracking.v1.TechProcessTracker import TechProcessTracker
from detection.pins_tracking.v1.VideoConfig import video6SolderConfig
from utils.VideoPlaybackHandlerBase import VideoPlaybackHandlerBase


class TechProcessVideoHandler(VideoPlaybackHandlerBase):
    def __init__(self, frameSize, framesDetections, videoWriter):
        super(TechProcessVideoHandler, self).__init__(frameSize)
        self.videoWriter = videoWriter
        self.framesDetections = framesDetections
        self.techProcessTracker = TechProcessTracker()
        self.frameDetections = None

    def frameReady(self, frame, framePos, framePosMsec, playback):
        self.frameDetections = self.framesDetections.get(framePos, [])
        self.techProcessTracker.track(self.frameDetections, framePos, framePosMsec, frame)
        super(TechProcessVideoHandler, self).frameReady(frame, framePos, framePosMsec, playback)

    def processDisplayFrame(self, displayFrame0):
        self.techProcessTracker.draw(displayFrame0)

        utils.visualize.drawDetections(displayFrame0, self.frameDetections)
        utils.visualize.putFramePos(displayFrame0, self._framePos, self._framePosMsec)
        self.techProcessTracker.drawStats(displayFrame0)

        self.videoWriter and self.videoWriter.write(displayFrame0)
        return displayFrame0

    def release(self):
        super(TechProcessVideoHandler, self).release()


def files():
    yield ('/HDD_DATA/Computer_Vision_Task/Video_6.mp4',
           '/HDD_DATA/Computer_Vision_Task/Video_6_result.mp4',
           DetectionsCSV.loadPickle('../../csv_cache/data/detections_video6.pcl'),
           video6SolderConfig)

    # yield ('/HDD_DATA/Computer_Vision_Task/Video_2.mp4',
    #        '/HDD_DATA/Computer_Vision_Task/Video_2_result.mp4',
    #        DetectionsCSV.loadPickle('../../csv_cache/data/detections_video2.pcl'),
    #        None)


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
    for sourceVideoFile, resultVideo, framesDetections, cfg in files():
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=False)
        videoWriter = None  # videoWriter(videoPlayback.cap, resultVideo)
        handler = TechProcessVideoHandler(videoPlayback.frameSize(), framesDetections, videoWriter)

        videoPlayback.playWithHandler(handler, range=getFramesRange())

        printMemoryUsage()

        cv2.waitKey()

        videoPlayback.release()
        handler.release()
        videoWriter and videoWriter.release()

    cv2.waitKey()


if __name__ == '__main__':
    main()
