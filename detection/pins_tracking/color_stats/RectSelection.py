import cv2

from utils import roundToInt
from utils.Geometry2D import Geometry2D


class RectSelection:
    def __init__(self, displayScale):
        self.displayScale = displayScale
        self.pt1 = self.pt2 = None

    def isPoint(self):
        return self.pt1 is not None and self.pt1 == self.pt2

    def selected(self):
        return self.pt1 is not None or self.pt2 is not None

    def displayPt(self, pt):
        x = roundToInt(pt[0] * self.displayScale)
        y = roundToInt(pt[1] * self.displayScale)
        return (x, y)

    def draw(self, img):
        if not self.selected():
            return img
        color = (0, 0, 200)
        thickness = 1
        if self.isPoint():
            thickness = 3
        return cv2.rectangle(img, self.displayPt(self.pt1), self.displayPt(self.pt2), color, thickness)

    def __normalizePoints(self):
        x1, y1 = self.pt1
        x2, y2 = self.pt2
        self.pt1 = (min(x1, x2), min(y1, y2))
        self.pt2 = (max(x1, x2), max(y1, y2))

    def __setPoint(self, displayX, displayY):
        originalX = roundToInt(displayX / self.displayScale)
        originalY = roundToInt(displayY / self.displayScale)
        pt = (originalX, originalY)
        if not self.selected():
            self.pt1 = self.pt2 = pt
        elif self.pt1 is not None and self.pt2 != self.pt1:
            # self.pt1 = self.pt2 = None
            return self.__resetNearestPoint(pt)
        elif self.pt1 is not None and self.pt2 == self.pt1:
            self.pt2 = pt
        self.__normalizePoints()
        return True

    def __resetNearestPoint(self, newPt):
        assert self.pt1 is not None and self.pt2 != self.pt1
        distToPt1 = Geometry2D.squaredL2Distance(self.pt1, newPt)
        distToPt2 = Geometry2D.squaredL2Distance(self.pt2, newPt)
        if distToPt1 == 0 or distToPt2 == 0:
            return False
        if distToPt1 < distToPt2:
            self.pt1 = newPt
        else:
            self.pt2 = newPt
        return True

    def __clear(self):
        if self.selected():
            self.pt1 = self.pt2 = None
            return True
        return False

    def mouseEvent(self, evt, displayX, displayY, flags, param):
        if evt == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY:
            return self.__setPoint(displayX, displayY)
        if evt == cv2.EVENT_LBUTTONDBLCLK:
            return self.__clear()
        return False
