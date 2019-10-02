import numpy as np
import cv2
import utils
from techprocess_tracking import DEBUG
from detection.PinDetector import PinDetector
from techprocess_tracking.SceneChanges import SceneChanges
from segmentation.SceneSegmentation import SceneSegmentation
from techprocess_tracking.StableScene import StableScene
from techprocess_tracking.TechProcessLogger import TechProcessLogger
from utils import visualize


class TechProcessTracker:
    def __init__(self, pinDetector: PinDetector, sceneSegmentation: SceneSegmentation):
        self.pinDetector = pinDetector
        self.sceneSegmentation = sceneSegmentation
        self.__stableScenes = []
        self.__currentScene = None
        self.sceneId = -1
        self.frameDetections = None

    def nextSceneId(self):
        self.sceneId += 1
        return self.sceneId

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

    def track(self, framePos, framePosMsec, frame):
        boxes, self.frameDetections = self.pinDetector.detect(frame, framePos, scoreThresh=.85)
        boxes = self.__skipEdgeBoxes(boxes, frame.shape)
        self.__trackBoxes(boxes, framePos, framePosMsec, frame)

    @staticmethod
    def __skipEdgeBoxes(boxes, frameShape):
        return [b for b in boxes if b.farFromFrameEdges(frameShape)]

    #####################################################################
    def __trackBoxes(self, bboxes, framePos, framePosMsec, frame):
        if not self.__currentScene:
            if any(bboxes):
                self.__currentScene = StableScene(bboxes, framePos, framePosMsec, frame, self.nextSceneId())
            else:
                # TODO:  collect background without pins and arms
                pass
            return

        lastStableScene = utils.lastOrDefault(self.__stableScenes, None)
        if lastStableScene:  # stable scene define workarea (cluster of pins)
            bboxes = lastStableScene.inWorkArea(bboxes)
            if len(bboxes) < lastStableScene.pinsCount:
                # CHECK - currently stabilized scene should be superset of lastStableScene
                self.__currentScene.finalize()
                self.__currentScene = StableScene(bboxes, framePos, framePosMsec, frame, self.nextSceneId())
                return

        currentSceneWasUnstable = not self.__currentScene.stabilized
        closeToCurrentScene = self.__currentScene.addIfClose(bboxes, framePos, framePosMsec, frame)

        if currentSceneWasUnstable and self.__currentScene.stabilized:
            # add to stable scene IF this scene was unstable before addition new frame and become stable after
            self.__registerCurrentSceneAsStable(frame, framePos)

        if not closeToCurrentScene:
            self.__currentScene.finalize()
            self.__currentScene = StableScene(bboxes, framePos, framePosMsec, frame, self.nextSceneId())

    ##############################################################################
    def __registerCurrentSceneAsStable(self, frame, framePos):
        assert self.__currentScene.stabilized
        assert self.__currentScene not in self.__stableScenes

        currentScene = self.__currentScene
        prevScene = utils.lastOrDefault(self.__stableScenes)
        sceneChanges = self.__registerSceneChanges(currentScene, prevScene, frame, framePos, self.sceneSegmentation)
        TechProcessLogger.logChanges(currentScene, sceneChanges)
        self.__stableScenes.append(currentScene)

    @staticmethod
    def __registerSceneChanges(currentScene: StableScene, prevScene: StableScene, frame, framePos,
                               sceneSegmentation: SceneSegmentation):
        assert currentScene.stabilized
        if prevScene is None:
            return SceneChanges(currentScene.pinsCount, 0)

        assert prevScene.stabilized
        assert currentScene.pinsCount >= prevScene.pinsCount

        if currentScene.pinsCount == prevScene.pinsCount:
            currentSceneSegmentation = sceneSegmentation.getSegmentationMap(frame, framePos)
            scaleY = frame.shape[0] / currentSceneSegmentation.shape[0]
            scaleX = frame.shape[1] / currentSceneSegmentation.shape[1]
            currentScene.detectSolder(prevScene, currentSceneSegmentation, scaleY, scaleX)

        pinsAdded = currentScene.pinsCount - prevScene.pinsCount
        solderAdded = currentScene.pinsWithSolderCount - prevScene.pinsWithSolderCount
        return SceneChanges(pinsAdded, solderAdded)

    def drawScene(self, img, withRawDetectionAndWorkArea=True):
        if withRawDetectionAndWorkArea:
            visualize.drawDetections(img, self.frameDetections)
        if self.__currentScene and self.__currentScene.stabilized:
            self.__currentScene.draw(img, withRawDetectionAndWorkArea)

    def getStats(self):
        if not any(self.__stableScenes):
            return 0, 0
        lastStableScene = self.__stableScenes[-1]
        return lastStableScene.pinsCount, lastStableScene.pinsWithSolderCount

    @staticmethod
    def DEBUG_show_diff_map(prevFrame_F32, currentFrame_F32):
        prev = cv2.cvtColor(prevFrame_F32, cv2.COLOR_BGR2GRAY)
        current = cv2.cvtColor(currentFrame_F32, cv2.COLOR_BGR2GRAY)
        diff = cv2.subtract(np.uint8(current), np.uint8(prev))
        DEBUG.imshow('scenes diff', np.uint8(diff))
