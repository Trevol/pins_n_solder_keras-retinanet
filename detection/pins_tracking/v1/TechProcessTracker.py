import numpy as np
import cv2
import utils
from detection.pins_tracking.v1.Box import Box
from detection.pins_tracking.v1.SceneChanges import SceneChanges
from detection.pins_tracking.v1.StableScene import StableScene


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
        changes = self.__registerSceneChanges(self.__currentScene, prevScene, self.sldConfig)
        self.__logNewStableChanges(self.__currentScene, changes)
        self.__stableScenes.append(self.__currentScene)
        self.__currentScene.stabilizedAtPos = self.__currentScene.lastFrame.pos

    @staticmethod
    def __registerSceneChanges(currentScene: StableScene, prevScene: StableScene, sldConfig):
        assert currentScene.stabilized
        if prevScene is None:
            return SceneChanges(currentScene.pinsCount, 0)

        assert prevScene.stabilized
        assert currentScene.pinsCount >= prevScene.pinsCount

        if currentScene.pinsCount == prevScene.pinsCount:
            currentScene.detectSolder(prevScene, sldConfig)
        pinsAdded = currentScene.pinsCount - prevScene.pinsCount
        solderAdded = currentScene.pinsWithSolderCount - prevScene.pinsWithSolderCount
        return SceneChanges(pinsAdded, solderAdded)

    @staticmethod
    def __logNewStableChanges(currentScene, changes):
        assert currentScene.stabilized
        if changes.pinsAdded == 0 and changes.solderAdded == 0:  # no changes - no log
            return
        framePos = currentScene.firstFrame.pos
        framePosMs = currentScene.firstFrame.posMsec
        pinsCount = currentScene.pinsCount
        pinsWithSolderCount = currentScene.pinsWithSolderCount
        logRecord = f'{framePos},{framePosMs:.0f},{pinsCount},{changes.pinsAdded},{pinsWithSolderCount},{changes.solderAdded}'
        print(logRecord)

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
