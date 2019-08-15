import numpy as np
import cv2
from utils.VideoPlayback import VideoPlayback
from utils.VideoPlaybackHandlerBase import VideoPlaybackHandlerBase


class MyVideoHandler(VideoPlaybackHandlerBase):
    def __init__(self, frameSize):
        super(MyVideoHandler, self).__init__(frameSize)
        self._frameScaleFactor = 1


def files():
    yield '/HDD_DATA/Computer_Vision_Task/Video_6.mp4'
    # yield '/HDD_DATA/Computer_Vision_Task/Video_2.mp4'


def main():
    for sourceVideoFile in files():
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=False)
        handler = MyVideoHandler(videoPlayback.frameSize())

        # framesRange = (4150, None)
        framesRange = None
        videoPlayback.play(range=framesRange, onFrameReady=handler.frameReady, onStateChange=handler.syncPlaybackState)
        videoPlayback.release()
        cv2.waitKey()
        handler.release()

    cv2.waitKey()


def save_frame_colors_at_point():
    def flipXY(*points):
        return [p[::-1] for p in points]

    list_yxOfInterest = flipXY((951, 186), (1233, 196), (1333, 186))
    frameColorsAtPoint = [[] for _ in list_yxOfInterest]

    for sourceVideoFile in files():
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=False)

        for pos, frame, _ in videoPlayback.frames():
            for index, yxOfInterest in enumerate(list_yxOfInterest):
                row = [pos, *frame[yxOfInterest]]
                frameColorsAtPoint[index].append(row)

        for yxOfInterest, colors in zip(list_yxOfInterest, frameColorsAtPoint):
            np.save(f'frame_colors_{yxOfInterest[1]}_{yxOfInterest[0]}.npy', colors)
        videoPlayback.release()


np.seterr(all='raise')
if __name__ == '__main__':
    main()
    # save_frame_colors_at_point()
