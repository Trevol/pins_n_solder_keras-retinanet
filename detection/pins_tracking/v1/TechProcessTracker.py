import numpy as np
import cv2
import utils
from detection.pins_tracking.v1.Box import Box
from detection.pins_tracking.v1.SceneChanges import SceneChanges
from detection.pins_tracking.v1.StableScene import StableScene
from detection.pins_tracking.v1.TechProcessLogger import TechProcessLogger


class TechProcessTracker:
    def __init__(self):
        self.__stableScenes = []
        self.__currentScene = None

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

    def lastStableScene(self):
        return utils.lastOrDefault(self.__stableScenes, None)

    def track(self, frameDetections, framePos, framePosMsec, frame):
        boxes = (Box(d[0]) for d in self.__skipWeakDetections(frameDetections))
        bboxes = self.__skipEdgeBoxes(boxes, frame.shape)
        self.__trackBoxes(bboxes, framePos, framePosMsec, frame)

    @staticmethod
    def __skipWeakDetections(detections):
        return (d for d in detections if d[2] >= .85)

    @staticmethod
    def __skipEdgeBoxes(boxes, frameShape):
        return [b for b in boxes if b.farFromFrameEdges(frameShape)]

    def __trackBoxes(self, bboxes, framePos, framePosMsec, frame):
        if not self.__currentScene:
            if any(bboxes):
                self.__currentScene = StableScene(bboxes, framePos, framePosMsec, frame)
            return

        lastStableScene = self.lastStableScene()
        if lastStableScene:
            bboxes = lastStableScene.inWorkArea(bboxes)
            if len(bboxes) < lastStableScene.pinsCount:
                # CHECK - currently stabilized scene should be superset of lastStableScene
                self.__currentScene = StableScene(bboxes, framePos, framePosMsec, frame)
                return

        currentSceneWasUnstable = not self.__currentScene.stabilized
        closeToCurrentScene = self.__currentScene.addIfClose(bboxes, framePos, framePosMsec, frame)

        if currentSceneWasUnstable and self.__currentScene.stabilized:
            # add to stable scene IF this scene was unstable before addition new frame and become stable after
            self.__registerCurrentSceneAsStable()

        if not closeToCurrentScene:
            self.__currentScene = StableScene(bboxes, framePos, framePosMsec, frame)

    def __registerCurrentSceneAsStable(self):
        assert self.__currentScene.stabilized
        assert self.__currentScene not in self.__stableScenes

        prevScene = utils.lastOrDefault(self.__stableScenes)
        sceneChanges = self.__registerSceneChanges(self.__currentScene, prevScene)
        TechProcessLogger.logChanges(self.__currentScene, sceneChanges)
        self.__stableScenes.append(self.__currentScene)
        self.__currentScene.stabilizedAtPos = self.__currentScene.lastFrame.pos

    @staticmethod
    def __registerSceneChanges(currentScene: StableScene, prevScene: StableScene):
        assert currentScene.stabilized
        if prevScene is None:
            return SceneChanges(currentScene.pinsCount, 0)

        assert prevScene.stabilized
        assert currentScene.pinsCount >= prevScene.pinsCount

        if currentScene.pinsCount == prevScene.pinsCount:
            currentScene.detectSolder(prevScene)
        pinsAdded = currentScene.pinsCount - prevScene.pinsCount
        solderAdded = currentScene.pinsWithSolderCount - prevScene.pinsWithSolderCount
        return SceneChanges(pinsAdded, solderAdded)

    def draw(self, img):
        if self.__currentScene and self.__currentScene.stabilized:
            self.__currentScene.draw(img)

    def drawStats(self, frame):
        if not any(self.__stableScenes):
            return
        lastStableScene = self.__stableScenes[-1]
        red = (0, 0, 255)
        text = f'Pins: {lastStableScene.pinsCount}'
        cv2.putText(frame, text, (10, 110), cv2.FONT_HERSHEY_COMPLEX, .7, red)
        text = f'Solder: {lastStableScene.pinsWithSolderCount}'
        cv2.putText(frame, text, (10, 140), cv2.FONT_HERSHEY_COMPLEX, .7, red)
