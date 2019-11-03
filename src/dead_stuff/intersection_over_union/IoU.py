import numpy as np
import cv2

from utils.Timer import timeit


class Box:
    class Union:
        pass

    class Intersection:
        def __init__(self, box1, box2):
            self.box1 = box1
            self.box2 = box2

        def vertices(self):
            result = set()
            """
            find all points of SELF inside OTHER
            find all points of OTHER inside SELF
            find points where SELF intersects with OTHER
                intersections of perpendicular sides (self.delatX != other.deltaX)  
            """
            box1 = self.box1
            box2 = self.box2
            for p in box1.vertices:
                if box2.containsPoint(p):
                    result.add(p)
            for p in box2.vertices:
                if box1.containsPoint(p):
                    result.add(p)
            for edge1 in box1.edges():
                for edge2 in box2.edges():
                    edgeIntesection = edge1.intersection(edge2)
                    if edgeIntesection:
                        result.add(edgeIntesection)
            return result

        def box(self):
            vertices = self.vertices()
            # TODO: analyze points and choose p00 and p11
            return None

        def area(self, other):
            box = self.box()
            if box is None:
                return 0
            box.area()

    def __init__(self, p0, p1):
        self.p00 = Point.convert(p0)
        self.p11 = Point.convert(p1)
        self.p01 = Point(self.p00.x, self.p11.y)
        self.p10 = Point(self.p11.x, self.p00.y)
        self.vertices = [self.p00, self.p11, self.p01, self.p10]

    def area(self):
        return (self.p11.x - self.p00.x) * (self.p11.y - self.p00.y)

    def edges(self):
        e00_10 = Edge(self.p00, self.p10)
        e10_11 = Edge(self.p10, self.p11)
        e00_01 = Edge(self.p00, self.p01)
        e01_11 = Edge(self.p01, self.p11)
        return [e00_10, e10_11, e00_01, e01_11]

    def containsPoint(self, p):
        return self.p00.x <= p.x <= self.p11.x and self.p00.y <= p.y <= self.p11.y

    def intersection(self, other):
        return self.Intersection(self, other)

    def union(self, other):
        return 1

    def iou(self, other):
        return self.intersection(self, other).area() / self.union(other).area()


class Edge:
    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1

    def intersection(self, other):
        edgesArePerpendicular = self.p0.x == self.p1.x and other.p0.y == other.p1.y or \
                                self.p0.y == self.p1.y and other.p0.x == other.p1.x
        if not edgesArePerpendicular:
            return None
        # x-coord of intersection: where p0.x == p1.x
        xCoord = self.p0.x if self.p0.x == self.p1.x else other.p0.x
        # y-coord of intersection: where p0.y == p1.y
        yCoord = self.p0.y if self.p0.y == self.p1.y else other.p0.y
        print(xCoord, yCoord)
        return Point(1, 1)


class Point:
    @classmethod
    def convert(cls, p):
        if isinstance(p, cls):
            return p
        return cls(p[0], p[1])

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self._hashable = (x, y)

    def __repr__(self):
        return str(self._hashable)

    def __hash__(self):
        return self._hashable.__hash__()

    def __eq__(self, other):
        # if not isinstance(other, Point): return False
        return self.x == other.x and self.y == other.y


def main():
    e1 = Edge(Point(1, 4), Point(6, 4))
    e2 = Edge(Point(5, 5), Point(5, 1))
    e1.intersection(e2)
    e2.intersection(e1)
    # b1 = Box((2, 1), (8, 6))
    # b2 = Box((5, -1), (7, 2))
    # print(b1.intersection(b2).vertices())
    # assert b1.intersection(b2) == 2


main()
