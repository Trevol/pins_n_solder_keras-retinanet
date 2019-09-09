import numpy as np
import cv2

from detection.csv_cache.DetectionsCSV import DetectionsCSV
import utils.visualize
from detection.pins_tracking.v2_motion_analysis.PinDetector import PickledDictionaryPinDetector
from utils.VideoPlayback import VideoPlayback

from utils.VideoPlaybackHandlerBase import VideoPlaybackHandlerBase


class TechProcessTracker_v2:
    def __init__(self, pinDetector):
        self.pinDetector = pinDetector

    def track(self, frame, framePos, framePosNsec):
        pass

    def draw(self, point, displayFrame):
        utils.visualize.drawDetections(displayFrame, self.frameDetections)
        self.drawStats(displayFrame)


class TechProcessVideoHandler_v2(VideoPlaybackHandlerBase):
    def __init__(self, frameSize, pinDetector):
        super(TechProcessVideoHandler_v2, self).__init__(frameSize)
        self.techProcessTracker = TechProcessTracker_v2(pinDetector)

    def frameReady(self, frame, framePos, framePosMsec, playback):
        self.techProcessTracker.track(frame, framePos, framePosMsec)
        super(TechProcessVideoHandler_v2, self).frameReady(frame, framePos, framePosMsec, playback)

    def processDisplayFrame(self, displayFrame0):
        utils.visualize.putFramePos((10, 40), displayFrame0, self._framePos, self._framePosMsec)
        self.techProcessTracker.draw((10, 110), displayFrame0)
        return displayFrame0

    def release(self):
        super(TechProcessVideoHandler_v2, self).release()


def files():
    yield ('/HDD_DATA/Computer_Vision_Task/Video_6.mp4',
           '/HDD_DATA/Computer_Vision_Task/Video_6_result.mp4',
           '../../csv_cache/data/detections_video6.pcl')

    # yield ('/HDD_DATA/Computer_Vision_Task/Video_2.mp4',
    #        '/HDD_DATA/Computer_Vision_Task/Video_2_result.mp4',
    #        DetectionsCSV.loadPickle('../../csv_cache/data/detections_video2.pcl'))


def main():
    np.seterr(all='raise')

    def getFramesRange():
        # framesRange = (4150, None)
        # framesRange = (8100, None)
        framesRange = None
        return framesRange

    for sourceVideoFile, resultVideo, detectionsPickledCache in files():
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=False)
        handler = TechProcessVideoHandler_v2(videoPlayback.frameSize(), PickledDictionaryPinDetector(detectionsPickledCache))

        endOfVideo = videoPlayback.playWithHandler(handler, range=getFramesRange())
        if endOfVideo:
            cv2.waitKey()

        videoPlayback.release()
        handler.release()


if __name__ == '__main__':
    main()
