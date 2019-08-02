from collections import deque
import numpy as np
import cv2
from more_itertools import ilen
from detection.pins_tracking.v1.Box import Box
from detection.pins_tracking.v1.Colors import Colors
from detection.pins_tracking.v1.Constants import StabilizationLength
from detection.pins_tracking.v1.Pin import Pin
from detection.pins_tracking.v1.FrameInfo import FrameInfo
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
        self.__pinsContour = None
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

    def detectSolder(self, prevScene, sldConfig):
        assert self.pinsCount == prevScene.pinsCount
        pinsAreClose, prevPins = self.__checkPinsCloseToScene(prevScene.pins)
        assert pinsAreClose
        for currentPin, prevPin in zip(self.__pins, prevPins):
            if prevPin.withSolder:
                currentPin.withSolder = prevPin.withSolder
            elif sldConfig and sldConfig.managePinSolder(currentPin, self.lastFrame.pos):
                pass
            else:
                currentPin.withSolder = currentPin.colorStat.areFromDifferentDistributions(prevPin.colorStat)
        #######################
        self.__pinsWithSolderCount = ilen(1 for p in self.__pins if p.withSolder)  # recompute count of pins with solder

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

    ####################################
    # TODO: refactor (remove code duplication) __checkPinsCloseToScene and __checkBoxesCloseToScene
    def __checkPinsCloseToScene(self, pins):
        assert self.__frames.notEmpty()
        if len(pins) != self.pinsCount:
            return False, None

        # TODO: store boxes/pins in (x, y) order to avoid constant sorting
        reorderedPins = []
        for currentPin in self.__pins:
            pinForCurrentPin = None
            maxDist = currentPin.box.cityblockDiagonal / 10  # TODO: make adaptive
            for pin in pins:
                if pin.box.withinDistance(currentPin.box, maxDist):
                    pinForCurrentPin = pin
                    reorderedPins.append(pinForCurrentPin)
                    break

            if not pinForCurrentPin:
                return False, None
        return True, reorderedPins

    def __checkBoxesCloseToScene(self, boxes):
        assert self.__frames.notEmpty()
        if len(boxes) != self.pinsCount:
            return False, None

        # TODO: store boxes/pins in (x, y) order to avoid constant sorting
        pinOrderedBoxes = []
        for pin in self.__pins:
            boxForPin = None
            maxDist = pin.box.cityblockDiagonal / 10  # TODO: make adaptive
            for box in boxes:
                if box.withinDistance(pin.box, maxDist):
                    boxForPin = box
                    pinOrderedBoxes.append(boxForPin)
                    break

            if not boxForPin:
                return False, None
        return True, pinOrderedBoxes

    #################################
    def __addToScene(self, frameInfo, frame):
        self.__frames.append(frameInfo)
        self.__updatePins(frame)

    def __updatePins(self, frame):
        assert self.__frames.notEmpty()

        if len(self.__frames.recent) == 1:
            boxes = self.__frames.first.bboxes
            meanColors = (self.__boxOuterMeanColor(frame, b) for b in boxes)
            self.__pins = [Pin(box, meanColor) for box, meanColor in zip(boxes, meanColors)]
            return

        for pinIndex in range(self.pinsCount):
            pinBoxesAcrossFrames = [frameInfo.bboxes[pinIndex] for frameInfo in self.__frames.recent]
            pinBox = Box.meanBox(pinBoxesAcrossFrames)
            meanColor = self.__boxOuterMeanColor(frame, pinBox)
            self.__pins[pinIndex].update(pinBox, meanColor)

        if self.stabilized:
            self.__measurePinsWorkArea()

    def __measurePinsWorkArea(self):
        boxes = [p.box.box for p in self.__pins]
        self.__pinsContour = Geometry2D.convexHull(boxes)

    @staticmethod
    def __boxOuterMeanColor(frame, innerBox):
        innerX0, innerY0, innerX1, innerY1 = innerBox.box
        dW, dH = innerBox.size / 4

        patch = frame[int(innerY0 - dH): int(innerY1 + dH + 1), int(innerX0 - dW): int(innerX1 + dW + 1)]
        patch = patch.astype(np.float32)

        # fill innerBox in path with NaN
        innerW, innerH = innerBox.size
        patch[int(dH):int(dH + innerH), int(dW):int(dW + innerW)] = np.NaN
        mean = np.nanmean(patch, (0, 1))
        return mean

    def draw(self, img):
        for pin in self.__pins:
            pin.draw(img)
        # TODO: when draw work area?
        if self.__pinsContour is not None:
            cv2.drawContours(img, [self.__pinsContour], 0, Colors.green, 1)

    def pinAtPoint(self, pt):
        pinsFilter = (p for p in self.pins if p.box.containsPoint(pt))
        return next(pinsFilter, None)
