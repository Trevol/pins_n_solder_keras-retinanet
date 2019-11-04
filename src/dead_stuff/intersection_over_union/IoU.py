import numpy as np
import cv2

from utils.Timer import timeit


class Box:
    class Union:
        def __init__(self, box1, box2):
            self.box1 = box1
            self.box2 = box2

        def area(self, intersectionArea):
            return self.box1.area() + self.box2.area() - intersectionArea

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
            if len(vertices) <= 1:
                return None
            assert len(vertices) == 4
            xx = [p.x for p in vertices]
            yy = [p.y for p in vertices]
            x0 = min(xx)
            y0 = min(yy)
            x1 = max(xx)
            y1 = max(yy)
            return Box(Point(x0, y0), Point(x1, y1))

        def area(self):
            box = self.box()
            if box is None:
                return 0
            return box.area()

    def __init__(self, p0, p1):
        p0 = Point.convert(p0)
        p1 = Point.convert(p1)
        assert p0.x < p1.x and p0.y < p1.y
        self.p00 = p0
        self.p11 = p1
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
        return self.Union(self, other)

    def iou(self, other):
        intersectionArea = self.intersection(other).area()
        return intersectionArea / self.union(other).area(intersectionArea)


class Edge:
    @staticmethod
    def __sort(p0, p1):
        if p0.x < p1.x or p0.y < p1.y:
            return p0, p1
        return p1, p0

    def __init__(self, p0, p1):
        self.p0, self.p1 = self.__sort(p0, p1)

    def intersection(self, other):
        edgesArePerpendicular = self.p0.x == self.p1.x and other.p0.y == other.p1.y or \
                                self.p0.y == self.p1.y and other.p0.x == other.p1.x
        if not edgesArePerpendicular:
            return

        if self.p0.x == self.p1.x:
            xEdge = self
            yEdge = other
        else:
            xEdge = other
            yEdge = self

        x = xEdge.p0.x
        y = yEdge.p0.y
        if x < yEdge.p0.x or x > yEdge.p1.x or y < xEdge.p0.y or y > xEdge.p1.y:  #
            return

        return Point(x, y)


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
        return other and self.x == other.x and self.y == other.y


def edgeIntersectionTest():
    e1 = Edge(Point(1, 4), Point(6, 4))
    e2 = Edge(Point(5, 5), Point(5, 1))
    assert e1.intersection(e2) == Point(5, 4)
    assert e2.intersection(e1) == Point(5, 4)

    e1 = Edge(Point(1, 4), Point(6, 4))
    e2 = Edge(Point(6, 5), Point(6, 1))
    assert e1.intersection(e2) == Point(6, 4)
    assert e2.intersection(e1) == Point(6, 4)

    e1 = Edge(Point(1, 4), Point(6, 4))
    e2 = Edge(Point(1, 5), Point(1, 1))
    assert e1.intersection(e2) == Point(1, 4)
    assert e2.intersection(e1) == Point(1, 4)

    e1 = Edge(Point(1, 4), Point(6, 4))
    e2 = Edge(Point(7, 5), Point(7, 1))
    assert e1.intersection(e2) is None
    assert e2.intersection(e1) is None

    e1 = Edge(Point(1, 4), Point(6, 4))
    e2 = Edge(Point(-1, 5), Point(-1, 1))
    assert e1.intersection(e2) is None
    assert e2.intersection(e1) is None


def main():
    b1 = Box((2, 1), (8, 6))
    b2 = Box((5, -1), (7, 2))
    assert b1.intersection(b2).vertices() == {Point(5, 1), Point(5, 2), Point(7, 2), Point(7, 1)}
    assert b1.intersection(b2).area() == 2
    assert b1.iou(b2) == 2 / 34


    b1 = Box((2, 1), (8, 6))
    b2 = Box((4, 3), (5, 4))
    assert b1.intersection(b2).vertices() == {Point(4, 3), Point(5, 4), Point(5, 3), Point(4, 4)}
    assert b1.intersection(b2).area() == 1
    assert b1.iou(b2) == 1 / 30

    b1 = Box((2, 1), (8, 6))
    b2 = Box((2, 5), (3, 6))
    assert b1.intersection(b2).area() == 1
    assert b1.iou(b2) == 1 / 30

    b1 = Box((2, 1), (8, 6))
    b2 = Box((1, 6), (2, 7))
    assert b1.intersection(b2).vertices() == {Point(2, 6)}
    assert b1.intersection(b2).area() == 0
    assert b1.iou(b2) == 0

    b1 = Box((2, 1), (8, 6))
    b2 = Box((1, 7), (2, 8))
    assert b1.intersection(b2).vertices() == set()
    assert b1.intersection(b2).area() == 0
    assert b1.iou(b2) == 0


main()
