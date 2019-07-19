import cv2
from utils.KbdKeys import KbdKeys


class VideoController2:
    def __init__(self, videoPlayback):
        self.videoPlayback = videoPlayback
        self._scheduleManualPlay = False

    @staticmethod
    def waitKey(delay):
        return cv2.waitKey(delay)

    def _handleManualPlay(self):
        assert self.videoPlayback.manualPlay

        read, stop, changed = False, False, False
        key = self.waitKey(-1)

        if key == KbdKeys.Q:
            self.videoPlayback.autoPlay = True
            read, stop, changed = True, False, True

        elif key == KbdKeys.ESC:
            read, stop, changed = False, True, False

        elif key == KbdKeys.L_ARROW:
            self.videoPlayback.backward()
            read, stop, changed = True, False, False

        elif key == KbdKeys.R_ARROW:
            read, stop, changed = True, False, False

        elif key == KbdKeys.UP_ARROW:
            self.videoPlayback.changeFrameDelay(+1)
            read, stop, changed = False, False, True

        elif key == KbdKeys.DOWN_ARROW:
            self.videoPlayback.changeFrameDelay(-1)
            read, stop, changed = False, False, True

        return read, stop, changed, key

    def _handleAutoPlay(self):
        assert self.videoPlayback.autoPlay

        read, stop, changed = False, False, False
        key = self.waitKey(self.videoPlayback.frameDelay)

        if key == -1:  # wait time elapsed
            read, stop, changed = True, False, False

        elif key == KbdKeys.ESC:
            read, stop, changed = False, True, False

        elif key == KbdKeys.Q:
            self.videoPlayback.autoPlay = False
            read, stop, changed = False, False, True

        elif key == KbdKeys.L_ARROW:
            self.videoPlayback.backward()
            # TODO: use writable properties autoPlay/manualPlay
            # TODO: use change state tracking at VideoPlayback - invoke stateChanged callback if state changed
            self.videoPlayback.autoPlay = False
            read, stop, changed = True, False, True

        elif key == KbdKeys.UP_ARROW:
            self.videoPlayback.changeFrameDelay(+1)
            read, stop, changed = False, False, True

        elif key == KbdKeys.DOWN_ARROW:
            self.videoPlayback.changeFrameDelay(-1)
            read, stop, changed = False, False, True

        return read, stop, changed, key

    def handleAction(self):
        if self.videoPlayback.manualPlay:
            return self._handleManualPlay()
        else:
            return self._handleAutoPlay()
