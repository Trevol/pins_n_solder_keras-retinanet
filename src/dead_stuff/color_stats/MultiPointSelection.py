import cv2

from utils.Colors import Colors
from utils import roundToInt
from utils import Geometry2D


class MultiPointSelection:
    def __init__(self, displayScale):
        self.displayScale = displayScale
        self.points = []

    def selected(self):
        return any(self.points)

    def empty(self):
        return not any(self.points)

    def draw(self, img):
        if self.empty():
            return
        thickness = 1
        for pt in self.points:
            cv2.rectangle(img, pt, (pt[0] + 1, pt[1] + 1), Colors.red, thickness)
            # cv2.rectangle(img, pt, pt, Colors.red, thickness)
        return img

    def __clear(self):
        if self.selected():
            self.points.clear()
            return True
        return False

    def __setPoint(self, displayX, displayY):
        originalX = roundToInt(displayX / self.displayScale)
        originalY = roundToInt(displayY / self.displayScale)
        eventPt = (originalX, originalY)
        if not self.selected():
            self.points.append(eventPt)
            return True
        nearestPt, nearestPtIndex, squaredDistance = self.__findNearestPoint(eventPt)
        if squaredDistance <= 100:
            del self.points[nearestPtIndex]
        else:
            self.points.append(eventPt)
        return True

    def __findNearestPoint(self, toPt):
        assert any(self.points)

        nearestPt = self.points[0]
        nearestPtIndex = 0
        minSquaredDistance = Geometry2D.squaredL2Distance(toPt, nearestPt)

        for index, pt in enumerate(self.points):
            squaredDist = Geometry2D.squaredL2Distance(toPt, nearestPt)
            if squaredDist < minSquaredDistance:
                nearestPt = pt
                nearestPtIndex = index
                minSquaredDistance = squaredDist
        return nearestPt, nearestPtIndex, minSquaredDistance


    def mouseEvent(self, evt, displayX, displayY, flags, param):
        if evt == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY:
            return self.__setPoint(displayX, displayY)
        if evt == cv2.EVENT_LBUTTONDBLCLK:
            return self.__clear()
        return False
