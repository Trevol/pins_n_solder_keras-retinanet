import cv2

from utils import roundToInt


class MultiPointSelection:
    def __init__(self, displayScale):
        self.displayScale = displayScale
        self.__points = []

    def selected(self):
        return any(self.__points)

    def draw(self, img):
        raise NotImplementedError()

    def __clear(self):
        if self.selected():
            self.__points.clear()
            return True
        return False

    def __setPoint(self, displayX, displayY):
        originalX = roundToInt(displayX / self.displayScale)
        originalY = roundToInt(displayY / self.displayScale)
        pt = (originalX, originalY)
        if not self.selected():
            self.__points.append(pt)
        elif self.pt1 is not None and self.pt2 != self.pt1:
            # self.pt1 = self.pt2 = None
            return self.__resetNearestPoint(pt)
        elif self.pt1 is not None and self.pt2 == self.pt1:
            self.pt2 = pt
        self.__normalizePoints()
        return True

    def mouseEvent(self, evt, displayX, displayY, flags, param):
        if evt == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY:
            return self.__setPoint(displayX, displayY)
        if evt == cv2.EVENT_LBUTTONDBLCLK:
            return self.__clear()
        return False
