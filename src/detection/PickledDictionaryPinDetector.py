from detection.Box import Box
from detection.PinDetector import PinDetector
from detection.csv_cache.DetectionsCSV import DetectionsCSV


class PickledDictionaryPinDetector(PinDetector):
    def __init__(self, pickleFile):
        self.__detectionsCache: dict = DetectionsCSV.loadPickle(pickleFile)

    def detect(self, frame, framePos, scoreThresh):
        allDetection = self.__detectionsCache.get(framePos, [])
        strongDetections = self.skipWeakDetections(allDetection, .85)
        boxes = (Box(d[0]) for d in strongDetections)
        return boxes, strongDetections