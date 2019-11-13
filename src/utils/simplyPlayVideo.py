import cv2
import utils.visualize
from utils.VideoPlaybackHandlerBase import VideoPlaybackHandlerBase
from utils.VideoPlayback import VideoPlayback


def playVideo():
    def files():
        yield '/HDD_DATA/Computer_Vision_Task/Video_6.mp4'
        # yield '/HDD_DATA/Computer_Vision_Task/Video_2.mp4'

    class VideoHandler(VideoPlaybackHandlerBase):
        def __init__(self, frameSize):
            super(VideoHandler, self).__init__(frameSize)
            self._frameScaleFactor = .85

    ################################################################
    for sourceVideoFile in files():
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=False)
        handler = VideoHandler(videoPlayback.frameSize())

        framesRange = (4000, None)
        # framesRange = None
        videoPlayback.playWithHandler(handler, framesRange)
        cv2.waitKey()
        videoPlayback.release()
        handler.release()

    cv2.waitKey()


###############################
playVideo()
