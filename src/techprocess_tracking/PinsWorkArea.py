import cv2

from models.detection import Box
from utils.Colors import Colors
from utils.Geometry2D import Geometry2D


class PinsWorkArea:
    def __init__(self, stableScenePins):
        assert any(stableScenePins)
        (self.__contour, self.__meanBoxSize, self.__minPinsDistance,
         self.__distToAreaThreshold) = self.__measure(stableScenePins)

    def draw(self, img, color=Colors.green):
        cv2.drawContours(img, [self.__contour], 0, color, 1)

    @classmethod
    def __measure(cls, stableScenePins):
        boxes = [p.box for p in stableScenePins]
        boxesBoxes = [b.box for b in boxes]
        contour = Geometry2D.convexHull(boxesBoxes)
        meanBoxSize = Box.meanSize(boxes)
        if len(stableScenePins) == 1:
            minPinsDistance = max(*boxes[0].size) * 3
        else:
            minPinsDistance = Geometry2D.minL2Distance([b.center for b in boxes])
        distToAreaThreshold = minPinsDistance * 1.5
        return contour, meanBoxSize, minPinsDistance, distToAreaThreshold

    def inWorkArea(self, boxes):
        boxesInArea = [b for b in boxes if self.__boxInWorkArea(b)]
        return boxesInArea

    def __distToArea(self, box):
        return cv2.pointPolygonTest(self.__contour, tuple(box.center), True)

    def __boxInWorkArea(self, box):
        return self.__distToArea(box) > -self.__distToAreaThreshold
