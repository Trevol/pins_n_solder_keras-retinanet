import numpy as np
import cv2
from utils import resize
from utils.VideoPlayback import VideoPlayback
import utils.Timer


class VideoHandler:
    winname = 'Video'

    def __init__(self, frameSize):
        self.__displayFrame = None
        cv2.namedWindow(self.winname)
        cv2.setMouseCallback(self.winname, self.onMouse)
        self.__frameScaleFactor = 1
        # self.__frameScaleFactor = 0.7 if frameSize[0] >= 1900 else 1

    def release(self):
        cv2.destroyWindow(self.winname)

    def onMouse(self, evt, x, y, flags, param):
        if evt == cv2.EVENT_LBUTTONUP and flags & cv2.EVENT_FLAG_CTRLKEY:
            if self.__displayFrame is not None:
                print((x, y), self.__displayFrame[y, x])

    def syncPlaybackState(self, frameDelay, autoPlay, framePos, framePosMsec, playback):
        autoplayLabel = 'ON' if autoPlay else 'OFF'
        stateTitle = f'{self.winname} (FrameDelay: {frameDelay}, Autoplay: {autoplayLabel})'
        cv2.setWindowTitle(self.winname, stateTitle)

    def frameReady(self, frame, framePos, framePosMsec, playback):
        self.__displayFrame = resize(frame, self.__frameScaleFactor)
        cv2.imshow(self.winname, self.__displayFrame)


def files():
    yield '/HDD_DATA/Computer_Vision_Task/Video_6.mp4'

    # yield '/HDD_DATA/Computer_Vision_Task/Video_2.mp4'


def main():
    for sourceVideoFile in files():
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=False)
        handler = VideoHandler(videoPlayback.frameSize())

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
        with utils.Timer.timeit():
            for pos, frame, _ in videoPlayback.frames():
                for index, yxOfInterest in enumerate(list_yxOfInterest):
                    row = [pos, *frame[yxOfInterest]]
                    frameColorsAtPoint[index].append(row)
        for yxOfInterest, colors in zip(list_yxOfInterest, frameColorsAtPoint):
            np.save(f'frame_colors_{yxOfInterest[1]}_{yxOfInterest[0]}.npy', colors)
        videoPlayback.release()


np.seterr(all='raise')
if __name__ == '__main__':
    save_frame_colors_at_point()
