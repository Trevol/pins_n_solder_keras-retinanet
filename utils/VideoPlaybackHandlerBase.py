import cv2

from utils import roundToInt, resize


class VideoPlaybackHandlerBase:
    winname = 'Video'

    def __init__(self, frameSize):
        self.__displayFrame0 = None
        self.__displayFrame = None
        self._frame = None
        self._framePos = None
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
        stateTitle = f'{self.winname} (FrameDelay: {frameDelay}, Autoplay: {autoplayLabel})'
        cv2.setWindowTitle(self.winname, stateTitle)

    def frameReady(self, frame, framePos, framePosMsec, playback):
        self._frame = frame
        self._framePos = framePos
        self.__displayFrame0 = self.createDisplayFrame()
        self.refreshDisplayFrame()

    def __showFrame(self):
        cv2.imshow(self.winname, self.__displayFrame)

    def refreshDisplayFrame(self):
        self.__displayFrame = self.processDisplayFrame(self.__displayFrame0)
        self.__showFrame()

    def createDisplayFrame(self):
        return resize(self._frame, self._frameScaleFactor)

    def processDisplayFrame(self, displayFrame0):
        # override this method to draw in display frame
        return displayFrame0
