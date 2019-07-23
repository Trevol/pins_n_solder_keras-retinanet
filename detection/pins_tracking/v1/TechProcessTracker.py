from detection.pins_tracking.v1.Box import Box
import cv2
from collections import deque


# TODO: calc bounding box for stable Scene -
class TechProcessTracker:
    def __init__(self):
        self.__stableRanges = []
        self.__currentScene = None

    def track(self, frameDetections, framePos, framePosMsec, frame):
        bboxes = [Box(d[0]) for d in frameDetections]
        self.__trackBoxes(bboxes, framePos, framePosMsec, frame)

    def __trackBoxes(self, bboxes, framePos, framePosMsec, frame):
        if not self.__currentScene:
            if any(bboxes):
                self.__currentScene = StableScene(bboxes, framePos, framePosMsec, frame)
            return

        wasUnstable = not self.__currentScene.stable

        closeToRange = self.__currentScene.addIfClose(bboxes, framePos, framePosMsec, frame)

        if wasUnstable and self.__currentScene.stable:
            # add to stable ranges IF range was unstable before addition new frame and become stable after
            self.__addCurrentRangeAsStable()

        if not closeToRange:
            self.__currentScene = StableScene(bboxes, framePos, framePosMsec, frame)

    def __addCurrentRangeAsStable(self):
        assert self.__currentScene.stable
        assert self.__currentScene not in self.__stableRanges

        self.__stableRanges.append(self.__currentScene)
        self.__logNewStableChanges()

    def __logNewStableChanges(self):
        assert any(self.__stableRanges)

        framePos, framePosMs, objectsCount = self.__stableRanges[-1].stats()
        print(f'{framePos}, {framePosMs:.0f}, {objectsCount}')

    def draw(self, img):
        if self.__currentScene and self.__currentScene.stable:
            self.__currentScene.draw(img)


class StableScene:
    class Frames:
        def __init__(self, stabilizationLenght):
            self.first = None
            self.recent = deque(maxlen=stabilizationLenght)

        def stabilized(self):
            return len(self.recent) == self.recent.maxlen

        def append(self, frameInfo):
            if not self.first:
                self.first = frameInfo
            self.recent.append(frameInfo)

        def notEmpty(self):
            return self.first is not None

    ##############################################
    __stabilizationLength = 5  # count of frames to ensure scene stability

    def __init__(self, bboxes, framePos, framePosMsec, frame):
        self.__frames = self.Frames(self.__stabilizationLength)
        # self.__frames_old = []  # or collections.deque(maxlen = self.__stabilizationCount)

        self.__instanceCount = None
        self.__meanBoxes = []
        self.addIfClose(bboxes, framePos, framePosMsec, frame)

    def stats(self):
        assert self.stable
        # first = self.__frames_old[0]
        # objectsCount = len(self.__meanBoxes)
        return self.__frames.first.pos, self.__frames.first.posMsec, self.__instanceCount

    @property
    def stable(self):
        # return len(self.__frames_old) >= self.__stabilizationLength
        return self.__frames.stabilized()

    def addIfClose(self, bboxes, framePos, framePosMsec, frame):
        if not any(bboxes):
            return False  # skip empty detections

        if not any(self.__frames.recent):
            self.__addToRange(FrameInfo(bboxes, framePos, framePosMsec, frame))
            return True  # first frame starts scene - so always belong to scene

        closeToRange, bboxes = self.__checkCloseToRange(bboxes)
        if closeToRange:
            self.__addToRange(FrameInfo(bboxes, framePos, framePosMsec, frame))
        return closeToRange

    ####################################
    def __checkCloseToRange(self, boxes):
        assert any(self.__meanBoxes)
        if len(boxes) < len(self.__meanBoxes):
            return False, None

        # detection close to mean boxes
        instanceOrderedBoxes = []
        for meanBox in self.__meanBoxes:
            boxForMeanBox = None
            maxDist = meanBox.cityblockDiagonal / 10  # TODO: make adaptive
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
            r = int(min(*meanBox.size) // 4)  # min(w,h)/4
            cv2.circle(img, tuple(meanBox.center), r, green, -1)

    def __addToRange(self, frameInfo):
        # self.__frames_old.append(frameInfo)
        self.__frames.append(frameInfo)
        self.__recalcStatistics()

    def __recalcStatistics(self):
        assert self.__frames.notEmpty()

        if len(self.__frames.recent) == 1:
            self.__meanBoxes = list(self.__frames.first.bboxes)
            self.__instanceCount = len(self.__meanBoxes)
            return

        for instanceIndex in range(self.__instanceCount):
            instanceBoxesAcrossFrames = [frameInfo.bboxes[instanceIndex] for frameInfo in self.__frames.recent]
            instanceMeanBox = Box.meanBox(instanceBoxesAcrossFrames)
            self.__meanBoxes[instanceIndex] = instanceMeanBox


#################################################
class FrameInfo:
    def __init__(self, bboxes, pos, posMsec, frame):
        self.pos = pos
        self.posMsec = posMsec
        self.bboxes = bboxes
        # TODO: extract frame patches for bboxes
