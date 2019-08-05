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

    def containsPoint(self, pt):
        return self.pt0[0] < pt[0] < self.pt1[0] and self.pt0[1] < pt[1] < self.pt1[1]

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
        raise NotImplementedError()
        sizes = []
        for box in boxes:
            sizes.append(box.size)
