from detection.Box import Box
from detection.csv_cache.DetectionsCSV import DetectionsCSV


class PinDetector:
    @staticmethod
    def skipWeakDetections(detections, scoreThreshold):
        return [d for d in detections if d[2] >= scoreThreshold]

    def detect(self, frame, framePos, scoreThresh):
        pass


class PickledDictionaryPinDetector(PinDetector):
    def __init__(self, pickleFile):
        self.__detectionsCache: dict = DetectionsCSV.loadPickle(pickleFile)

    def detect(self, frame, framePos, scoreThresh):
        allDetection = self.__detectionsCache.get(framePos, [])
        strongDetections = self.skipWeakDetections(allDetection, .85)
        boxes = (Box(d[0]) for d in strongDetections)
        return boxes, strongDetections


