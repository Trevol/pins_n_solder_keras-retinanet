from detection.csv_cache.DetectionsCSV import DetectionsCSV


class PinDetector:
    def detect(self, frame, framePos):
        pass


class PickledDictionaryPinDetector(PinDetector):
    def __init__(self, pickleFile):
        self.__detectionsCache: dict = DetectionsCSV.loadPickle(pickleFile)

    def detect(self, frame, framePos):
        return self.__detectionsCache.get(framePos, [])


class NNPinDetector(PinDetector):
    def detect(self, frame, framePos):
        pass
