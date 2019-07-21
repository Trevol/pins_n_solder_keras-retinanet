class TechProcessTracker:

    def __init__(self):
        self.__stableRanges = []
        self.__currentFrameRange = None

    def track(self, rawDetections, framePos, frameMs, frame):
        if not self.__currentFrameRange:
            self.__currentFrameRange = FrameRange(rawDetections, framePos, frameMs, frame)
            return

        wasUnstable = not self.__currentFrameRange.stable

        belongToRange = self.__currentFrameRange.tryAdd(rawDetections, framePos, frameMs, frame)

        if wasUnstable and self.__currentFrameRange.stable:  # add to stable ranges IF unstable before addition new frame and become stable after
            self.__addCurrentRangeAsStable()

        if not belongToRange:  # start new FrameRange
            self.__currentFrameRange = FrameRange(rawDetections, framePos, frameMs, frame)
            pass

    def __addCurrentRangeAsStable(self):
        assert self.__currentFrameRange.stable
        assert self.__currentFrameRange not in self.__stableRanges

        self.__stableRanges.append(self.__currentFrameRange)
        self.logNewStableChanges()

    def logNewStableChanges(self):
        raise NotImplementedError()

    def draw(self, img):
        if any(self.__stableRanges):  # draw last stable range
            self.__stableRanges[-1].draw(img)


class FrameRange:
    def __init__(self, rawDetections, framePos, frameMs, frame):
        self.__frames = []  # or collections.deque(maxlen = 50)
        self.__stable = False
        self.tryAdd(rawDetections, framePos, frameMs, frame)

    @property
    def stable(self):
        return self.__stable

    def tryAdd(self, rawDetections, framePos, frameMs, frame):
        frameInfo = FrameInfo(rawDetections, framePos, frameMs, frame)
        belongToRange = True
        if not any(self.__frames):
            self.__frames.append(frameInfo)
            return belongToRange  # True

        belongToRange = self.__checkBelongingToRange(frameInfo, self.__frames)
        if belongToRange:
            self.__addToRange(frameInfo)

        return belongToRange

    @staticmethod
    def __checkBelongingToRange(frameInfo, frameInfos):
        # TODO: check actual belonging to Range
        raise NotImplementedError()
        pass

    def draw(self, img):
        raise NotImplementedError()

    def __addToRange(self, frameInfo):
        raise NotImplementedError
        self.__frames.append(frameInfo)
        # recalc aggregated values - max/avg bbox/center
        pass


class FrameInfo:
    def __init__(self, rawDetections, framePos, frameMs, frame):
        pass
