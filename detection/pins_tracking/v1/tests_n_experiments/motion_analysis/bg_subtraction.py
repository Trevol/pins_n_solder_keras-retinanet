from utils.VideoPlayback import VideoPlayback
from utils.VideoPlaybackHandlerBase import VideoPlaybackHandlerBase


def files():
    yield '/HDD_DATA/Computer_Vision_Task/Video_6.mp4'
    # yield '/HDD_DATA/Computer_Vision_Task/Video_2.mp4'


class VideoHandler(VideoPlaybackHandlerBase):
    pass


def main():
    for sourceVideoFile in files():
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=False)
        handler = VideoHandler(videoPlayback.frameSize())
        videoPlayback.playWithHandler(handler)

        videoPlayback.release()
        handler.release()


#########################
#########################
main()
