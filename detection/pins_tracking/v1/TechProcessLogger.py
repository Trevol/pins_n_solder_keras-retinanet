class TechProcessLogger:
    @staticmethod
    def logChanges(currentScene, sceneChanges):
        assert currentScene.stabilized
        if sceneChanges.pinsAdded == 0 and sceneChanges.solderAdded == 0:  # no changes - no log
            return
        framePos = currentScene.firstFrame.pos
        framePosMs = currentScene.firstFrame.posMsec
        pinsCount = currentScene.pinsCount
        pinsWithSolderCount = currentScene.pinsWithSolderCount
        logRecord = f'{framePos},{framePosMs:.0f},{pinsCount},{sceneChanges.pinsAdded},{pinsWithSolderCount},{sceneChanges.solderAdded}'
        print(logRecord)