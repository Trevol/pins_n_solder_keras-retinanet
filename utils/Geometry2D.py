import math

import cv2
import numpy as np


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