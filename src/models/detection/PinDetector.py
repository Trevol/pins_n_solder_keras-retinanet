class PinDetector:
    @staticmethod
    def skipWeakDetections(detections, scoreThreshold):
        return [d for d in detections if d[2] >= scoreThreshold]

    def detect(self, frame, framePos, scoreThresh):
        pass


