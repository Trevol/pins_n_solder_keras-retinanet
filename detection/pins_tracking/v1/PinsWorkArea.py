import cv2

from detection.pins_tracking.v1.Colors import Colors
from utils.Geometry2D import Geometry2D


class PinsWorkArea:
    def __init__(self, stableScenePins):
        assert any(stableScenePins)
        self.__measure(stableScenePins)

    def draw(self, img, color=Colors.green):
        cv2.drawContours(img, [self.__contour], 0, color, 1)

    def __measure(self, stableScenePins):
        boxes = [p.box.box for p in stableScenePins]
        self.__contour = Geometry2D.convexHull(boxes)
        # self.__meanBoxSizeWH = Box.meanSize(boxes)

    def filterOutsiderBoxes(self, boxes):
        # TODO: filter boxes which are to far to workArea
        pass

    def inWorkArea(self, boxes):
        return boxes

    @staticmethod
    def inWorkBox(box, workBox):
        wx0, wy0, wx1, wy1 = workBox
        x0, y0, x1, y1 = box
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        # box center in workBox
        return wx0 < cx < wx1 and wy0 < cy < wy1
