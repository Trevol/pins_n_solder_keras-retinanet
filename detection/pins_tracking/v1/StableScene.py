import warnings
from collections import deque
import numpy as np
import cv2
from more_itertools import ilen

from detection.pins_tracking.v1 import DEBUG
from detection.pins_tracking.v1.Box import Box
from detection.pins_tracking.v1.Colors import Colors
from detection.pins_tracking.v1.Constants import StabilizationLength
from detection.pins_tracking.v1.Pin import Pin
from detection.pins_tracking.v1.FrameInfo import FrameInfo
from detection.pins_tracking.v1.PinsWorkArea import PinsWorkArea
from detection.pins_tracking.v1.segmentation.classesMeta import BGR
from detection.pins_tracking.v1.segmentation.pin_utils import colorizeLabel
from utils.Geometry2D import Geometry2D

framesCounter = 0


class StableScene:
    class FrameInfosQueue:
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

    def __init__(self, bboxes, framePos, framePosMsec, frame, sceneId):
        self.__sceneId = sceneId
        self.__frameInfos = self.FrameInfosQueue()
        self.__framesBuffer = deque(maxlen=StabilizationLength)
        self.__framesBuffer.append(frame)
        self.aggregatedFrame_F32 = None
        self.__aggregatedAtFramePos = None
        self.__pins = []
        self.__pinsWorkArea = None
        self.__pinsWithSolderCount = 0
        self.stabilizedAtPos = None
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
    def firstFrameInfo(self):
        return self.__frameInfos.first

    @property
    def lastFrameInfo(self):
        return self.__frameInfos.recent[-1]

    @property
    def stabilized(self):
        return self.__frameInfos.stabilized()

    def addIfClose(self, bboxes, framePos, framePosMsec, frame):
        if not any(bboxes):
            return False  # skip empty detections

        if not any(self.__frameInfos.recent):
            self.__addToScene(FrameInfo(bboxes, framePos, framePosMsec, frame), frame)
            return True  # first frame starts scene - so always belong to scene

        closeToScene, bboxes = self.__checkBoxesCloseToScene(bboxes)
        if closeToScene:
            self.__addToScene(FrameInfo(bboxes, framePos, framePosMsec, frame), frame)
        return closeToScene

    def finalize(self):
        if self.stabilized:
            self.__aggregateFrames()
        self.__framesBuffer.clear()
        self.__framesBuffer = None

    def __aggregateFrames(self):
        assert self.stabilized
        currentFramePos = self.lastFrameInfo.pos
        if self.__aggregatedAtFramePos == currentFramePos:
            return
        bufferLen = len(self.__framesBuffer)
        # извлекаем поливину буфера из его середины
        framesBuffer = list(self.__framesBuffer)[bufferLen // 4:3 * (bufferLen // 4)]
        self.aggregatedFrame_F32 = self.__meanFrame(framesBuffer)
        self.__aggregatedAtFramePos = currentFramePos
        # DEBUG.imshow(f'aggregatedFrame {self.__sceneId}', self.aggregatedFrame_F32.astype(np.uint8))

    @staticmethod
    def __meanFrame(framesBuffer):
        accum = np.float32(framesBuffer[0])
        for frame in framesBuffer[1:]:
            cv2.accumulate(frame, accum)
        mean = np.divide(accum, len(framesBuffer), out=accum)
        return mean

    def __checkPinsCloseToScene(self, pins):
        boxes = [p.box for p in pins]
        return self.__checkBoxesCloseToScene(boxes, pins)

    def __checkBoxesCloseToScene(self, boxes, boxedObjects=None):
        assert self.__frameInfos.notEmpty()
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
        self.__frameInfos.append(frameInfo)
        self.__framesBuffer.append(frame)
        if self.stabilized and self.stabilizedAtPos is None:
            self.__aggregateFrames()
            self.stabilizedAtPos = frameInfo.pos
        self.__updatePins(frame)

    def __updatePins(self, frame):
        assert self.__frameInfos.notEmpty()

        if len(self.__frameInfos.recent) == 1:
            boxes = self.__frameInfos.first.bboxes
            self.__pins = [Pin(box) for box in boxes]
            return

        boxes = []
        for pinIndex in range(self.pinsCount):
            pinBoxesAcrossFrames = [frameInfo.bboxes[pinIndex] for frameInfo in self.__frameInfos.recent]
            pinBox = Box.meanBox(pinBoxesAcrossFrames)
            boxes.append(boxes)
            self.__pins[pinIndex].update(pinBox)
        if self.stabilized:
            self.__pinsWorkArea = PinsWorkArea(self.__pins)

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

    def detectSolder(self, prevScene, currentSceneSegmentation, sceneSegmentationScaleY, sceneSegmentationScaleX):
        assert self.pinsCount == prevScene.pinsCount
        pinsAreClose, prevPins = self.__checkPinsCloseToScene(prevScene.pins)
        assert pinsAreClose

        # TODO: need scale to downscale pins boxes to currentSceneSegmentation
        # sceneSegmentationScale = frame.shape[0] / currentSceneSegmentation.shape[0]

        def DEBUG():
            img = colorizeLabel(currentSceneSegmentation, BGR)
            for pin in self.__pins:
                rescaledBox = pin.box.rescale(sceneSegmentationScaleY, sceneSegmentationScaleX)
                x0, y0, x1, y1 = rescaledBox.box
                cv2.rectangle(img, (x0, y0), (x1, y1), 255, 1)
            cv2.imshow('DEBUG GGG', img)

        DEBUG()

        for currentPin, prevPin in zip(self.__pins, prevPins):
            if prevPin.withSolder:
                currentPin.withSolder = prevPin.withSolder
            else:
                currentPin.withSolder = self.__detectSolderOnPin(currentPin, currentSceneSegmentation,
                                                                 sceneSegmentationScaleY, sceneSegmentationScaleX)

        self.__pinsWithSolderCount = ilen(1 for p in self.__pins if p.withSolder)  # recompute count of pins with solder

    @staticmethod
    def __detectSolderOnPin(pin, currentSceneSegmentation, sceneSegmentationScaleY, sceneSegmentationScaleX):
        pinWithSolderLabel = 2
        # project pin box to sceneSegmentation
        rescaledBox = pin.box.rescale(sceneSegmentationScaleY, sceneSegmentationScaleX)
        x0, y0, x1, y1 = rescaledBox.box
        boxOnMap = currentSceneSegmentation[y0:y1, x0:x1]
        solderMask = np.equal(boxOnMap, pinWithSolderLabel).astype(np.uint8)
        solderArea = cv2.countNonZero(solderMask)
        totalArea = (x1 - x0) * (y1 - y0)

        # count pixels inside box with pinWithSolderLabel and compare with box area
        return (solderArea / totalArea) > .5

# @staticmethod
# def __boxOuterMeanColor(frame, innerBox):
#     innerX0, innerY0, innerX1, innerY1 = innerBox.box
#     # protect against boxes near frame edges
#     assert innerX0 > 0 and innerY0 > 0 and innerX1 < frame.shape[1] - 1 and innerY1 < frame.shape[0] - 1
#     dW, dH = innerBox.size / 4
#
#     patch = frame[int(innerY0 - dH): int(innerY1 + dH + 1), int(innerX0 - dW): int(innerX1 + dW + 1)]
#     patch = patch.astype(np.float32)
#
#     # fill innerBox in path with NaN
#     innerW, innerH = innerBox.size
#     patch[int(dH):int(dH + innerH), int(dW):int(dW + innerW)] = np.NaN
#     mean = np.nanmean(patch, (0, 1))
#     return mean
