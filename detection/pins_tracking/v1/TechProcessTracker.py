from detection.pins_tracking.v1.Box import Box
import cv2


class TechProcessTracker:
    def __init__(self):
        self.__stableRanges = []
        self.__currentFrameRange = None

    def track(self, frameDetections, framePos, framePosMsec, frame):
        bboxes = [Box(d[0]) for d in frameDetections]
        self.__trackBoxes(bboxes, framePos, framePosMsec, frame)

    def __trackBoxes(self, bboxes, framePos, framePosMsec, frame):
        if not self.__currentFrameRange:
            if any(bboxes):
                self.__currentFrameRange = FrameRange(bboxes, framePos, framePosMsec, frame)
            return

        wasUnstable = not self.__currentFrameRange.stable

        closeToRange = self.__currentFrameRange.addIfClose(bboxes, framePos, framePosMsec, frame)

        if wasUnstable and self.__currentFrameRange.stable:
            # add to stable ranges IF range was unstable before addition new frame and become stable after
            self.__addCurrentRangeAsStable()

        if not closeToRange:  # start new FrameRange
            self.__currentFrameRange = FrameRange(bboxes, framePos, framePosMsec, frame)

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
    def __init__(self, bboxes, framePos, frameMs, frame):
        self.__frames = []  # or collections.deque(maxlen = 50)
        self.__meanBoxes = []
        self.addIfClose(bboxes, framePos, frameMs, frame)

    @property
    def stable(self):
        return len(self.__frames) >= 3

    def addIfClose(self, bboxes, framePos, framePosMsec, frame):
        if not any(bboxes):
            return False  # skip empty detections

        if not any(self.__frames):
            self.__addToRange(FrameInfo(bboxes, framePos, framePosMsec, frame))
            return True  # first frame belong to Range

        closeToRange, bboxes = self.__checkCloseToRange(bboxes)
        if closeToRange:
            self.__addToRange(FrameInfo(bboxes, framePos, framePosMsec, frame))
        return closeToRange

    ####################################
    def __checkCloseToRange(self, boxes):
        assert any(self.__meanBoxes)
        if len(boxes) != len(self.__meanBoxes):
            return False, None

        # detection close to mean boxes
        instanceOrderedBoxes = []
        for meanBox in self.__meanBoxes:
            boxForMeanBox = None
            maxDist = meanBox.cityblockDiagonal / 8  # box.distToCenter / 4
            for box in boxes:
                if box.withinDistance(meanBox, maxDist):
                    boxForMeanBox = box
                    instanceOrderedBoxes.append(boxForMeanBox)
                    break

            if not boxForMeanBox:
                return False, None
        return True, instanceOrderedBoxes

    #################################

    def draw(self, img):
        green = (0, 200, 0)
        for meanBox in self.__meanBoxes:
            cv2.circle(img, tuple(meanBox.center), 3, green, -1)


    def __addToRange(self, frameInfo):
        self.__frames.append(frameInfo)
        self.__recalcStatistics()

    def __recalcStatistics(self):
        assert any(self.__frames)

        if len(self.__frames) == 1:
            self.__meanBoxes = list(self.__frames[0].bboxes)
            return

        countOfInstances = len(self.__frames[0].bboxes)
        for instanceIndex in range(countOfInstances):
            instanceBoxesAcrossFrames = [frameInfo.bboxes[instanceIndex] for frameInfo in self.__frames]
            instanceMeanBox = Box.meanBox(instanceBoxesAcrossFrames)
            self.__meanBoxes[instanceIndex] = instanceMeanBox


class FrameInfo:
    def __init__(self, bboxes, pos, posMsec, frame):
        self.pos = pos
        self.posMsec = posMsec
        self.bboxes = bboxes
        # TODO: extract frame patches for bboxes
