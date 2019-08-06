import math
import cv2
import numpy as np
import itertools


class Geometry2D:
    @staticmethod
    def boxPoints(box):
        x0 = math.floor(box[0])
        y0 = math.floor(box[1])
        x1 = math.ceil(box[2])
        y1 = math.ceil(box[3])

        topLeft = x0, y0
        yield topLeft
        bottomRight = x1, y1
        yield bottomRight
        bottomLeft = x0, y1
        yield bottomLeft
        topRight = x1, y0
        yield topRight

    @staticmethod
    def convexHull(boxes):
        pts = np.int32([pt for b in boxes for pt in Geometry2D.boxPoints(b)])
        return cv2.convexHull(pts)

    pointCombinationStruct = np.dtype('2f4,2f4')

    @classmethod
    def pairwiseL2Distances(cls, pts):
        # http://numpy-discussion.10968.n7.nabble.com/itertools-combinations-to-numpy-td16635.html
        pointsCombinations = np.fromiter(itertools.combinations(pts, 2), cls.pointCombinationStruct)
        pointsCombinations = pointsCombinations.view(np.float32).reshape(-1, 4)
        vectors = np.subtract(pointsCombinations[:, :2], pointsCombinations[:, 2:])
        x = vectors[:, 0]
        y = vectors[:, 1]
        distances = np.sqrt(x * x + y * y)
        return distances

    @classmethod
    def minL2Distance(cls, pts):
        return cls.pairwiseL2Distances(pts).min()
