import numpy as np
import cv2

from utils.Timer import timeit


class Box:
    @staticmethod
    def toPoint(p):
        if isinstance(p, Point):
            return p
        return Point(p[0], p[1])

    def __init__(self, p0, p1):
        self.p00 = self.toPoint(p0)
        self.p11 = self.toPoint(p1)

    def area(self):
        return (self.p11.x - self.p00.x) * (self.p11.y - self.p00.y)

    def points(self):
        p01 = Point(self.p00.x, self.p11.y)
        p10 = Point(self.p11.x, self.p00.y)
        return [self.p00, self.p11, p01, p10]

    def containsPoint(self, p):
        return self.p00.x <= p.x <= self.p11.x and self.p00.y <= p.y <= self.p11.y

    def intersectionPoints(self, other):
        """Area of intersection"""
        result = set()
        """
        find all points of SELF inside OTHER
        find all points of OTHER inside SELF
        find points where SELF intersects with OTHER
            intersections of perpendicular sides (self.delatX != other.deltaX)  
        """
        for p in self.points():
            if other.containsPoint(p):
                result.add(p)
        for p in other.points():
            if self.containsPoint(p):
                result.add(p)

        return result

    def intersectionBox(self, other):
        points = self.intersectionPoints()
        # TODO: analize points
        return None

    def intersectionArea(self, other):
        box = self.intersectionBox()
        if box is None:
            return 0
        box.area()

    def union(self, other):
        return 1

    def iou(self, other):
        return self.intersection(other) / self.union(other)


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self._hashable = (x, y)

    def __repr__(self):
        return str(self._hashable)

    def __hash__(self):
        return self._hashable.__hash__()


def main():
    b1 = Box((2, 1), (8, 6))
    b2 = Box((5, -1), (7, 2))
    print(b1.intersectionPoints(b2))
    # assert b1.intersection(b2) == 2


main()
