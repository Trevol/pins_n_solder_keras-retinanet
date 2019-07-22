class TechProcessTracker:
    def __init__(self):
        self.__stableRanges = []
        self.__currentFrameRange = None

    def track(self, frameDetections, framePos, framePosMsec, frame):
        if not self.__currentFrameRange:
            if any(frameDetections):
                self.__currentFrameRange = FrameRange(frameDetections, framePos, framePosMsec, frame)
            return

        wasUnstable = not self.__currentFrameRange.stable

        belongToRange = self.__currentFrameRange.tryAdd(frameDetections, framePos, framePosMsec, frame)

        if wasUnstable and self.__currentFrameRange.stable:
            # add to stable ranges IF range was unstable before addition new frame and become stable after
            self.__addCurrentRangeAsStable()

        if not belongToRange:  # start new FrameRange
            self.__currentFrameRange = FrameRange(frameDetections, framePos, framePosMsec, frame)

    def __addCurrentRangeAsStable(self):
        assert self.__currentFrameRange.stable
        assert self.__currentFrameRange not in self.__stableRanges

        self.__stableRanges.append(self.__currentFrameRange)
        self.__logNewStableChanges()

    def __logNewStableChanges(self):
        # raise NotImplementedError()
        # diff between
        pass

    def draw(self, img):
        if any(self.__stableRanges):  # draw last stable range
            self.__stableRanges[-1].draw(img)


class FrameRange:
    def __init__(self, frameDetections, framePos, frameMs, frame):
        self.__frames = []  # or collections.deque(maxlen = 50)
        self.tryAdd(frameDetections, framePos, frameMs, frame)
        self.__statistics = None

    @property
    def stable(self):
        return len(self.__frames) >= 3

    def tryAdd(self, frameDetections, framePos, framePosMsec, frame):
        if not any(frameDetections):
            return False  # skip empty detections

        if not any(self.__frames):
            self.__addToRange(FrameInfo(frameDetections, framePos, framePosMsec, frame))
            return True  # first frame belong to Range

        belongToRange, frameDetections = self.__checkBelongingToRange(frameDetections)
        if belongToRange:
            self.__addToRange(FrameInfo(frameDetections, framePos, framePosMsec, frame))
        return belongToRange

    def __checkBelongingToRange(self, detections):
        assert any(self.__frames)
        if len(detections) != self.__statistics.detectionsCount:
            return False
        for d in detections:
            bbox = d[0]

        raise NotImplementedError()

    def draw(self, img):
        raise NotImplementedError()

    def __addToRange(self, frameInfo):
        self.__frames.append(frameInfo)
        self.__recalcStatistics()

    def __recalcStatistics(self):
        # recalc aggregated values - max/avg bbox/center
        raise NotImplementedError('Recalc ')


class FrameInfo:
    def __init__(self, detections, pos, posMsec, frame):
        self.pos = pos
        self.posMsec = posMsec
        self.bboxes = [d[0] for d in detections]
        # TODO: extract frame patches for bboxes
