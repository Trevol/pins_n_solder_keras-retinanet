import warnings
from collections import deque
import numpy as np
import cv2
from more_itertools import ilen
from detection.pins_tracking.v1.Box import Box
from detection.pins_tracking.v1.Colors import Colors
from detection.pins_tracking.v1.Constants import StabilizationLength
from detection.pins_tracking.v1.Pin import Pin
from detection.pins_tracking.v1.FrameInfo import FrameInfo
from detection.pins_tracking.v1.PinsWorkArea import PinsWorkArea
from utils.Geometry2D import Geometry2D


class StableScene:
    class Frames:
        def __init__(self):
            self.first = None
            self.recent = deque(maxlen=StabilizationLength)

        def stabilized(self):
            return len(self.recent) == self.recent.maxlen

        def append(self, frameInfo):
            if not self.first:
                self.first = frameInfo
            self.recent.append(frameInfo)

        def notEmpty(self):
            return self.first is not None

    ##############################################

    def __init__(self, bboxes, framePos, framePosMsec, frame):
        self.__frames = self.Frames()
        self.__pins = []
        self.__pinsWorkArea = None
        self.__pinsWithSolderCount = 0
        self.addIfClose(bboxes, framePos, framePosMsec, frame)

    @property
    def pins(self):
        return tuple(self.__pins)

    @property
    def pinsCount(self):
        return len(self.__pins)

    @property
    def pinsWithSolderCount(self):
        return self.__pinsWithSolderCount  # len([p for p in self.__pins if p.withSolder])

    @property
    def firstFrame(self):
        return self.__frames.first

    @property
    def lastFrame(self):
        return self.__frames.recent[-1]

    @property
    def stabilized(self):
        return self.__frames.stabilized()

    def detectSolder(self, prevScene):
        # TODO: implement it using pins masks and analyzing differences between stable scenes
        # assert self.pinsCount == prevScene.pinsCount
        # pinsAreClose, prevPins = self.__checkPinsCloseToScene(prevScene.pins)
        # assert pinsAreClose
        #
        # for currentPin, prevPin in zip(self.__pins, prevPins):
        #     if prevPin.withSolder:
        #         currentPin.withSolder = prevPin.withSolder
        #     else:
        #         currentPin.withSolder = currentPin.colorStat.areFromDifferentDistributions(prevPin.colorStat)
        #
        # self.__pinsWithSolderCount = ilen(1 for p in self.__pins if p.withSolder)  # recompute count of pins with solder
        pass

    def addIfClose(self, bboxes, framePos, framePosMsec, frame):
        if not any(bboxes):
            return False  # skip empty detections

        if not any(self.__frames.recent):
            self.__addToScene(FrameInfo(bboxes, framePos, framePosMsec, frame), frame)
            return True  # first frame starts scene - so always belong to scene

        closeToScene, bboxes = self.__checkBoxesCloseToScene(bboxes)
        if closeToScene:
            self.__addToScene(FrameInfo(bboxes, framePos, framePosMsec, frame), frame)
        return closeToScene

    def __checkPinsCloseToScene(self, pins):
        boxes = [p.box for p in pins]
        return self.__checkBoxesCloseToScene(boxes, pins)

    def __checkBoxesCloseToScene(self, boxes, boxedObjects=None):
        assert self.__frames.notEmpty()
        if len(boxes) != self.pinsCount:
            return False, None

        # TODO: store boxes/pins in (x, y)-order to avoid constant sorting
        if boxedObjects is None:
            boxedObjects = boxes

        pinOrderedBoxedObjects = []
        for currentPin in self.__pins:
            boxedObjectsCloseToCurrentPin = None
            maxDist = currentPin.box.cityblockDiagonal / 10  # TODO: make adaptive
            for box, boxedObject in zip(boxes, boxedObjects):
                if box.withinDistance(currentPin.box, maxDist):
                    boxedObjectsCloseToCurrentPin = boxedObject
                    pinOrderedBoxedObjects.append(boxedObjectsCloseToCurrentPin)
                    break

            if not boxedObjectsCloseToCurrentPin:
                return False, None
        return True, pinOrderedBoxedObjects

    def __addToScene(self, frameInfo, frame):
        self.__frames.append(frameInfo)
        self.__updatePins(frame)

    def __updatePins(self, frame):
        assert self.__frames.notEmpty()

        if len(self.__frames.recent) == 1:
            boxes = self.__frames.first.bboxes
            self.__pins = [Pin(box) for box in boxes]
            return

        boxes = []
        for pinIndex in range(self.pinsCount):
            pinBoxesAcrossFrames = [frameInfo.bboxes[pinIndex] for frameInfo in self.__frames.recent]
            pinBox = Box.meanBox(pinBoxesAcrossFrames)
            boxes.append(boxes)
            self.__pins[pinIndex].update(pinBox)
        if self.stabilized:
            self.__pinsWorkArea = PinsWorkArea(self.__pins)

    @staticmethod
    def __boxOuterMeanColor(frame, innerBox):
        innerX0, innerY0, innerX1, innerY1 = innerBox.box
        # protect against boxes near frame edges
        assert innerX0 > 0 and innerY0 > 0 and innerX1 < frame.shape[1] - 1 and innerY1 < frame.shape[0] - 1
        dW, dH = innerBox.size / 4

        patch = frame[int(innerY0 - dH): int(innerY1 + dH + 1), int(innerX0 - dW): int(innerX1 + dW + 1)]
        patch = patch.astype(np.float32)

        # fill innerBox in path with NaN
        innerW, innerH = innerBox.size
        patch[int(dH):int(dH + innerH), int(dW):int(dW + innerW)] = np.NaN
        mean = np.nanmean(patch, (0, 1))
        return mean

    def inWorkArea(self, boxes):
        assert self.__pinsWorkArea
        return self.__pinsWorkArea.inWorkArea(boxes)

    def draw(self, img):
        for pin in self.__pins:
            pin.draw(img)
        if self.__pinsWorkArea:
            self.__pinsWorkArea.draw(img)

    def pinAtPoint(self, pt):
        pinsFilter = (p for p in self.pins if p.box.containsPoint(pt))
        return next(pinsFilter, None)
