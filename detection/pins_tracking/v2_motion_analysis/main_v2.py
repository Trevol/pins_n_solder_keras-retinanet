import numpy as np
import cv2

from detection.csv_cache.DetectionsCSV import DetectionsCSV
import utils.visualize
from utils.VideoPlayback import VideoPlayback

from utils.VideoPlaybackHandlerBase import VideoPlaybackHandlerBase


class TechProcessTracker_v2:
    def track(self, frame, framePos, framePosNsec):
        pass


class TechProcessVideoHandler_v2(VideoPlaybackHandlerBase):
    def __init__(self, frameSize, framesDetections, videoWriter):
        super(TechProcessVideoHandler_v2, self).__init__(frameSize)
        self.videoWriter = videoWriter
        self.framesDetections = framesDetections
        self.techProcessTracker = TechProcessTracker_v2()
        self.frameDetections = None

    def frameReady(self, frame, framePos, framePosMsec, playback):
        self.frameDetections = self.framesDetections.get(framePos, [])
        self.techProcessTracker.track(self.frameDetections, framePos, framePosMsec, frame)
        super(TechProcessVideoHandler_v2, self).frameReady(frame, framePos, framePosMsec, playback)

    def processDisplayFrame(self, displayFrame0):
        self.techProcessTracker.draw(displayFrame0)

        utils.visualize.drawDetections(displayFrame0, self.frameDetections)
        utils.visualize.putFramePos(displayFrame0, self._framePos, self._framePosMsec)
        self.techProcessTracker.drawStats(displayFrame0)

        self.videoWriter and self.videoWriter.write(displayFrame0)
        return displayFrame0

    def release(self):
        super(TechProcessVideoHandler_v2, self).release()


def files():
    yield ('/HDD_DATA/Computer_Vision_Task/Video_6.mp4',
           '/HDD_DATA/Computer_Vision_Task/Video_6_result.mp4',
           DetectionsCSV.loadPickle('../../csv_cache/data/detections_video6.pcl'))

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

    for sourceVideoFile, resultVideo, framesDetections in files():
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=False)
        handler = TechProcessVideoHandler_v2(videoPlayback.frameSize(), framesDetections)

        endOfVideo = videoPlayback.playWithHandler(handler, range=getFramesRange())
        if endOfVideo:
            cv2.waitKey()

        videoPlayback.release()
        handler.release()


if __name__ == '__main__':
    main()
