import cv2

from detection.pins_tracking.v1.Box import Box
from detection.pins_tracking.v1.Colors import Colors
from utils.Geometry2D import Geometry2D


class PinsWorkArea:
    def __init__(self, stableScenePins):
        assert any(stableScenePins)
        self.__measure(stableScenePins)

    def draw(self, img, color=Colors.green):
        cv2.drawContours(img, [self.__contour], 0, color, 1)

    def __measure(self, stableScenePins):
        # TODO: if we have 1 pin????
        if len(stableScenePins) == 1:  # TODO: if we have 1 pin????
            raise Exception('len(stableScenePins) == 1')
        boxes = [p.box for p in stableScenePins]
        rawBoxes = [p.box.box for p in stableScenePins]
        self.__contour = Geometry2D.convexHull(rawBoxes)
        self.__meanBoxSize = Box.meanSize(boxes)
        self.minPinsDistance = self.__computeMinPinsDistance(stableScenePins)

    @staticmethod
    def __computeMinPinsDistance(stableScenePins):
        centers = [p.box.center for p in stableScenePins]
        minDist = Geometry2D.minL2Distance(centers)
        return minDist

    def inWorkArea(self, boxes):
        centers = [tuple(b.center) for b in boxes]
        return boxes

    # @staticmethod
    # def inWorkBox(box, workBox):
    #     wx0, wy0, wx1, wy1 = workBox
    #     x0, y0, x1, y1 = box
    #     cx = (x0 + x1) / 2
    #     cy = (y0 + y1) / 2
    #     # box center in workBox
    #     return wx0 < cx < wx1 and wy0 < cy < wy1
