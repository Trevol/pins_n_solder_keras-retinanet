import numpy as np

from utils import boxCenter, cityblockDistance


class Box:
    def __init__(self, bbox):
        self.box = bbox
        self.pt0 = np.array([bbox[0], bbox[1]])
        self.pt1 = np.array([bbox[2], bbox[3]])
        self.size = np.abs(self.pt1 - self.pt0)
        self.center = boxCenter(bbox)
        self.cityblockDiagonal = cityblockDistance(self.pt0, self.pt1)

    def rescale(self, scaleY, scaleX):
        x0 = self.box[0] // scaleX
        y0 = self.box[1] // scaleY
        x1 = self.box[2] // scaleX
        y1 = self.box[3] // scaleY
        box = np.array([x0, y0, x1, y1], np.int32)
        return Box(box)

    def containsPoint(self, pt):
        return self.pt0[0] < pt[0] < self.pt1[0] and self.pt0[1] < pt[1] < self.pt1[1]

    def farFromFrameEdges(self, frameShape):
        maxY = frameShape[0] - 1
        maxX = frameShape[1] - 1
        dw, dh = self.size / 4
        x0 = self.pt0[0]
        y0 = self.pt0[1]
        x1 = self.pt1[0]
        y1 = self.pt1[1]
        return x0 - dw > 0 and y0 - dw > 0 and x1 + dw < maxX and y1 + dh < maxY

    @staticmethod
    def boxByPoint(boxes, pt):
        boxes = (b for b in boxes if b.containsPoint(pt))
        return next(boxes, None)

    def distance(self, otherBox):
        # L1/Manhattan/cityblock distance for simplicity
        return cityblockDistance(self.center, otherBox.center)

    def withinDistance(self, otherBox, maxDistance):
        return self.distance(otherBox) <= maxDistance

    def nearest(self, otherBoxes):
        assert any(otherBoxes)
        minDistance = np.inf
        nearestResult = None
        for indx, otherBox in enumerate(otherBoxes):
            dist = self.distance(otherBox)
            if dist < minDistance:
                minDistance = dist
                nearestResult = indx, otherBox, dist
        return nearestResult

    @staticmethod
    def meanBox(boxes):
        pts0 = [b.pt0 for b in boxes]
        pts1 = [b.pt1 for b in boxes]
        meanPt0 = np.mean(pts0, axis=0)
        meanPt1 = np.mean(pts1, axis=0)
        meanBox = np.hstack((meanPt0, meanPt1))
        return Box(meanBox)

    @staticmethod
    def meanSize(boxes):
        sizes = [b.size for b in boxes]
        return np.mean(sizes, 0)
