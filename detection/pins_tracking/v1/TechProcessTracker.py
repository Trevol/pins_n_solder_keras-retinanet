import numpy as np
from detection.pins_tracking.v1.Box import Box
import cv2
from collections import deque

# count of frames to ensure scene stability
from detection.pins_tracking.v1.Constants import StabilizationLength
from detection.pins_tracking.v1.Pin import Pin


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
        self.addIfClose(bboxes, framePos, framePosMsec, frame)

    @property
    def pins(self):
        return tuple(self.__pins)

    @property
    def pinsCount(self):
        return len(self.__pins)

    @property
    def pinsWithSolderCount(self):
        # TODO: how to count generated items without list creation
        return len([p for p in self.__pins if p.withSolder])

    @property
    def firstFrame(self):
        return self.__frames.first

    @property
    def lastFrame(self):
        return self.__frames.recent[-1]

    @property
    def stable(self):
        return self.__frames.stabilized()

    def detectSolder(self, prevScene, sldConfig):
        assert self.pinsCount == prevScene.pinsCount
        pinsAreClose, prevPins = self.__checkPinsCloseToScene(prevScene.pins)
        assert pinsAreClose
        for currentPin, prevPin in zip(self.__pins, prevPins):

            if prevPin.withSolder:
                currentPin.withSolder = prevPin.withSolder
                continue

            done = False
            if sldConfig is not None and any(sldConfig):
                for pt, shouldStabilizeAtPos in sldConfig:
                    if currentPin.box.containsPoint(pt):
                        if self.lastFrame.pos == shouldStabilizeAtPos:
                            currentPin.withSolder = True
                        done = True
                        break
            if done:
                continue

            currentPin.withSolder = self.__colorsAreFromDifferentDistributions(currentPin.colorStat, prevPin.colorStat)

    @staticmethod
    def __colorsAreFromDifferentDistributions(colorStat1, colorStat2):
        # absdiff = np.abs(colorStat1.mean - colorStat2.mean)
        # thresh = colorStat1.std * 3
        absdiff = np.abs(colorStat1.median - colorStat2.median)
        colorsSoDifferent = np.any(absdiff > 14.5)  # at least one component is outside of std deviation
        return colorsSoDifferent

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

    def pinAtPoint(self, pt):
        pinsFilter = (p for p in self.pins if p.box.containsPoint(pt))
        return next(pinsFilter, None)


# TODO: calc bounding box for stable Scene -
class TechProcessTracker:
    def __init__(self, sldConfig):
        self.__stableScenes = []
        self.__currentScene = None
        self.sldConfig = sldConfig

    def dumpPinStats(self, pt):
        if not any(self.__stableScenes):
            return
        currentScene = self.__stableScenes[-1]
        print((pt, currentScene.stabilizedAtPos))
        return

        if not any(self.__stableScenes):
            return
        currentScene = self.__stableScenes[-1]
        pinAtCurrentScene = currentScene.pinAtPoint(pt)
        if not pinAtCurrentScene:
            return

        pinAtPrevScene = None
        if len(self.__stableScenes) >= 2:
            prevScene = self.__stableScenes[-2]
            pinAtPrevScene = prevScene.pinAtPoint(pt)
        self.__dump(pinAtCurrentScene, pinAtPrevScene)

    def __dump(self, pinAtCurrent, pinAtPrev):
        assert pinAtCurrent

        print('')
        print('----------------------------------------------------')
        if not pinAtPrev:
            currentStat = pinAtCurrent.colorStat
            print('CURRENT:')
            print(f'mean:{np.round(currentStat.mean, 1)}')
            print(f'std:{np.round(currentStat.std, 1)}')
            print(f'median: {np.round(currentStat.median, 1)}')
            for color in pinAtCurrent.meanColors:
                print(' ', np.round(color, 1))
        else:
            currentStat = pinAtCurrent.colorStat
            prevStat = pinAtPrev.colorStat
            print('CURRENT/PREV:')
            print(f'mean:{np.round(currentStat.mean, 1)} / {np.round(prevStat.mean, 1)}')
            print(f'std:{np.round(currentStat.std, 1)} / {np.round(prevStat.std, 1)}')
            print(f'median: {np.round(currentStat.median, 1)} / {np.round(prevStat.median, 1)}')
            for currentColor, prevColor in zip(pinAtCurrent.meanColors, pinAtPrev.meanColors):
                print(' ', np.round(currentColor, 1), np.round(prevColor, 1))

    def track(self, frameDetections, framePos, framePosMsec, frame):
        bboxes = [Box(d[0]) for d in frameDetections]
        self.__trackBoxes(bboxes, framePos, framePosMsec, frame)

    def __trackBoxes(self, bboxes, framePos, framePosMsec, frame):
        if not self.__currentScene:
            if any(bboxes):
                self.__currentScene = StableScene(bboxes, framePos, framePosMsec, frame)
            return

        if any(self.__stableScenes) and len(bboxes) < self.__stableScenes[-1].pinsCount:
            # CHECK - currently stabilized scene should be superset of self.__stableScenes[-1]
            self.__currentScene = StableScene(bboxes, framePos, framePosMsec, frame)
            return

        currentSceneWasUnstable = not self.__currentScene.stable
        closeToCurrentScene = self.__currentScene.addIfClose(bboxes, framePos, framePosMsec, frame)

        if currentSceneWasUnstable and self.__currentScene.stable:
            # add to stable scene IF this scene was unstable before addition new frame and become stable after
            self.__registerCurrentSceneAsStable()

        if not closeToCurrentScene:
            self.__currentScene = StableScene(bboxes, framePos, framePosMsec, frame)

    def __registerCurrentSceneAsStable(self):
        assert self.__currentScene.stable
        assert self.__currentScene not in self.__stableScenes

        prevScene = self.__stableScenes[-1] if any(self.__stableScenes) else None
        changes = self.__registerSceneChanges(self.__currentScene, prevScene, self.sldConfig)
        self.__logNewStableChanges(self.__currentScene, changes)
        self.__stableScenes.append(self.__currentScene)
        self.__currentScene.stabilizedAtPos = self.__currentScene.lastFrame.pos

    @staticmethod
    def __registerSceneChanges(currentScene: StableScene, prevScene: StableScene, sldConfig):
        assert currentScene.stable
        if prevScene is None:
            return SceneChanges(currentScene.pinsCount, 0)

        assert prevScene.stable
        assert currentScene.pinsCount >= prevScene.pinsCount
        # TODO: detect solder
        if currentScene.pinsCount == prevScene.pinsCount:
            currentScene.detectSolder(prevScene, sldConfig)
        pinsAdded = currentScene.pinsCount - prevScene.pinsCount
        solderAdded = currentScene.pinsWithSolderCount - prevScene.pinsWithSolderCount
        return SceneChanges(pinsAdded, solderAdded)

    @staticmethod
    def __logNewStableChanges(currentScene, changes):
        assert currentScene.stable
        if changes.pinsAdded == 0 and changes.solderAdded == 0:  # no changes - no log
            return
        framePos = currentScene.firstFrame.pos
        framePosMs = currentScene.firstFrame.posMsec
        pinsCount = currentScene.pinsCount
        pinsWithSolderCount = currentScene.pinsWithSolderCount
        logRecord = f'{framePos},{framePosMs:.0f},{pinsCount},{changes.pinsAdded},{pinsWithSolderCount},{changes.solderAdded}'
        print(logRecord)

    def draw(self, img):
        if self.__currentScene and self.__currentScene.stable:
            self.__currentScene.draw(img)

    def drawStats(self, frame):
        # if not self.__currentScene or not self.__currentScene.stable:
        #     return
        if not any(self.__stableScenes):
            return
        lastStableScene = self.__stableScenes[-1]
        red = (0, 0, 255)
        text = f'Pins: {lastStableScene.pinsCount}'
        cv2.putText(frame, text, (10, 110), cv2.FONT_HERSHEY_COMPLEX, .7, red)
        text = f'Solder: {lastStableScene.pinsWithSolderCount}'
        cv2.putText(frame, text, (10, 140), cv2.FONT_HERSHEY_COMPLEX, .7, red)


class SceneChanges:
    def __init__(self, pinsAdded, solderAdded):
        self.pinsAdded = pinsAdded
        self.solderAdded = solderAdded


#################################################
class FrameInfo:
    def __init__(self, bboxes, pos, posMsec, frame):
        self.pos = pos
        self.posMsec = posMsec
        self.bboxes = bboxes
        # TODO: extract frame patches for bboxes
        # self.framePatch = self.__extractPatches(frame, bboxes)

    def boxByPoint(self, pt):
        return Box.boxByPoint(self.bboxes, pt)

    @classmethod
    def __extractPatches(cls, frame, bboxes):
        return [cls.__extractPatch(frame, b) for b in bboxes]

    @staticmethod
    def __extractPatch(frame, bbox):
        patchBox = np.ceil(bbox.box).astype(np.int32) + 2
        x0, y0, x1, y1 = patchBox
        patch = frame[y0:y1, x0:x1].copy()
        return patch
