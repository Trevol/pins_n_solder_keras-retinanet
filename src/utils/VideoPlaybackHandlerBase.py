import cv2

import utils
from utils import roundToInt, resize


class VideoPlaybackHandlerBase:
    winname = 'Video'

    def __init__(self, frameSize):
        self.__displayFrame0 = None
        self.__displayFrame = None
        self._frame = None
        self._framePos = None
        self._framePosMsec = None
        cv2.namedWindow(self.winname)
        cv2.setMouseCallback(self.winname, self.onMouse)
        self._frameScaleFactor = 0.7 if frameSize[0] >= 1900 else 1

    def release(self):
        cv2.destroyWindow(self.winname)
        self._displayFrame = None
        self._frame = None

    def onMouse(self, evt, displayFrameX, displayFrameY, flags, param):
        if evt == cv2.EVENT_LBUTTONUP and flags & cv2.EVENT_FLAG_CTRLKEY:
            if self._frame is not None:
                originalX = roundToInt(displayFrameX / self._frameScaleFactor)
                originalY = roundToInt(displayFrameY / self._frameScaleFactor)
                color = self._frame[originalY, originalX]
                print('Original Coords:', (originalX, originalY), 'Color:', color, 'Display Coords:',
                      (displayFrameX, displayFrameY))

    def syncPlaybackState(self, frameDelay, autoPlay, framePos, framePosMsec, playback):
        autoplayLabel = 'ON' if autoPlay else 'OFF'
        stateTitle = f'Video (Pos: {framePos}/{framePosMsec:.1f}ms, FrameDelay: {frameDelay}, Autoplay: {autoplayLabel})'
        cv2.setWindowTitle(self.winname, stateTitle)

    def frameReady(self, frame, framePos, framePosMsec, playback):
        self._frame = frame
        self._framePos = framePos
        self._framePosMsec = framePosMsec
        self.__displayFrame0 = self.createDisplayFrame()
        self.refreshDisplayFrame()

    def __showFrame(self):
        cv2.imshow(self.winname, self.__displayFrame)

    def refreshDisplayFrame(self):
        displayFrame = self.processDisplayFrame(self.__displayFrame0)
        self.__displayFrame = resize(displayFrame, self._frameScaleFactor)
        self.__showFrame()

    def createDisplayFrame(self):
        # separate original frame from drawings
        return self._frame.copy()

    def processDisplayFrame(self, displayFrame0):
        # override this method to draw in display frame
        return displayFrame0