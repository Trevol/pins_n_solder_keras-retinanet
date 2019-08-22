import cv2

from utils import roundToInt


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

    def addDisplayPoint(self, displayX, displayY):
        originalX = roundToInt(displayX / self.displayScale)
        originalY = roundToInt(displayY / self.displayScale)
        pt = (originalX, originalY)
        if self.pt1 is None and self.pt2 is None:
            self.pt1 = self.pt2 = pt
        elif self.pt1 is not None and self.pt2 != self.pt1:
            self.pt1 = self.pt2 = None
        elif self.pt1 is not None and self.pt2 == self.pt1:
            self.pt2 = pt
            x1, y1 = self.pt1
            x2, y2 = self.pt2
            self.pt1 = (min(x1, x2), min(y1, y2))
            self.pt2 = (max(x1, x2), max(y1, y2))
