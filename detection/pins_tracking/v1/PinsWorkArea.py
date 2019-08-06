import cv2

from detection.pins_tracking.v1.Box import Box
from detection.pins_tracking.v1.Colors import Colors
from utils.Geometry2D import Geometry2D
from utils.Timer import timeit


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
        self.__minPinsDistance = self.__computeMinPinsDistance(stableScenePins)
        self.__distToAreaThreshold = self.__minPinsDistance * 1.5

    @staticmethod
    def __computeMinPinsDistance(stableScenePins):
        centers = [p.box.center for p in stableScenePins]
        minDist = Geometry2D.minL2Distance(centers)
        return minDist

    def inWorkArea(self, boxes):
        boxesInArea = [b for b in boxes if self.__boxInWorkArea(b)]
        return boxesInArea

    def __distToArea(self, box):
        return cv2.pointPolygonTest(self.__contour, tuple(box.center), True)

    def __boxInWorkArea(self, box):
        return self.__distToArea(box) > -self.__distToAreaThreshold
