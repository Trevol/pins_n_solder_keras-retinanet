import numpy as np
import cv2
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

    def frameReady(self, frame, framePos, framePosMsec, playback):
        frameDetections = self.framesDetections.get(framePos, [])
        self.techProcessTracker.track(frameDetections, framePos, framePosMsec, frame)

        self.techProcessTracker.draw(frame)

        utils.visualize.drawDetections(frame, frameDetections)
        utils.visualize.putFramePos(frame, framePos, framePosMsec)
        self.techProcessTracker.drawStats(frame)

        self.videoWriter and self.videoWriter.write(frame)

        super(TechProcessVideoHandler, self).frameReady(frame, framePos, framePosMsec, playback)

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


def main():
    np.seterr(all='raise')
    for sourceVideoFile, resultVideo, framesDetections, cfg in files():
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=False)
        videoWriter = None  # videoWriter(videoPlayback.cap, resultVideo)
        handler = TechProcessVideoHandler(videoPlayback.frameSize(), framesDetections, videoWriter)

        framesRange = (4150, None)
        # framesRange = (8100, None)
        # framesRange = None
        videoPlayback.play(range=framesRange, onFrameReady=handler.frameReady, onStateChange=handler.syncPlaybackState)
        videoPlayback.release()
        cv2.waitKey()
        handler.release()
        videoWriter and videoWriter.release()

    cv2.waitKey()


if __name__ == '__main__':
    main()
