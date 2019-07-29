# coords for video_6
class SolderConfig:
    def __init__(self, data):
        self.__data = data

    def managePinSolder(self, pin, framePos):
        if not any(self.__data):
            return False
        # return pinManagedBySldConfig, pinHasSolderBySldConfig
        for pt, shouldAppearsAtPos in self.__data:
            if pin.box.containsPoint(pt):
                pin.withSolder = framePos == shouldAppearsAtPos
                return True
        return False


video6SolderConfig = SolderConfig([
    # (point, framePos)
    ((423, 706), 5359),
    ((470, 510), 5980),
    ((1443, 329), 7216),
    ((1346, 324), 7409),
    ((1240, 327), 7409),
    ((1149, 329), 7445),
    ((1436, 243), 7820),
    ((1329, 241), 7820),
    ((1237, 243), 8013),
    ((1139, 249), 8013),
    ((860, 240), 8113),
    ((766, 246), 8141),
    ((680, 241), 8235),
    ((587, 244), 8235),
    ((501, 250), 8262),
    ((1426, 156), 8299),
    ((1316, 156), 8380),
    ((1223, 167), 8380),
    ((1127, 161), 8457),
    ((679, 243), 8235),
    ((950, 161), 8457000),
    ((857, 164), 8457000)
])
