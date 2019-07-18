import cv2
from utils.KbdKeys import KbdKeys


class VideoPlaybackActionResult:
    ChangeState = 1
    NextFrame = 2
    Break = 3


class VideoController2:
    def __init__(self, videoPlayback):
        self.videoPlayback = videoPlayback
        self._scheduleManualPlay = False

    def _enterManualPlay(self):
        assert self.videoPlayback.autoPlay

        self._scheduleManualPlay = False
        self.videoPlayback.autoPlay = False
        return self._inManualPlay()

    def _inManualPlay(self):
        assert self.videoPlayback.manualPlay

        action = None
        key = cv2.waitKey(-1)

        if key == KbdKeys.Q:
            self._enterAutoPlay()
            action = VideoPlaybackActionResult.ChangeState
        elif key == KbdKeys.ESC:
            action = VideoPlaybackActionResult.Break
        elif key == KbdKeys.L_ARROW:
            self.videoPlayback.backward()
            action = VideoPlaybackActionResult.NextFrame
        elif key == KbdKeys.R_ARROW:
            action = VideoPlaybackActionResult.NextFrame
        elif key == KbdKeys.UP_ARROW:
            self.videoPlayback.changeFrameDelay(+1)
            action = VideoPlaybackActionResult.ChangeState
        elif key == KbdKeys.DOWN_ARROW:
            self.videoPlayback.changeFrameDelay(-1)
            action = VideoPlaybackActionResult.ChangeState

        return key, action

    def _enterAutoPlay(self):
        assert self.videoPlayback.manualPlay
        self.videoPlayback.autoPlay = True

    def _inAutoPlay(self):
        assert self.videoPlayback.autoPlay

        action = None
        key = cv2.waitKey(self.videoPlayback.frameDelay)

        if key == KbdKeys.Q:
            return self._enterManualPlay()
        elif key == KbdKeys.L_ARROW:
            self.videoPlayback.backward()
            self._schedulePause = True  # return outside for frame retrieval and then enterPause
        elif key == KbdKeys.UP_ARROW:
            self._speedUp()
        elif key == KbdKeys.DOWN_ARROW:
            self._speedDown()

        return key, action

    def handleAction(self):
        if self._scheduleManualPlay:
            return self._enterManualPlay()
        if self.videoPlayback.manualPlay:
            return self._inManualPlay()
        else:
            return self._inAutoPlay()

# class VideoController_OLD:
#     class States:
#         PAUSE = 'pause'
#         PLAYING = 'playing'
#
#     def __init__(self, delay, state=None):
#         self.delay = delay
#         self.state = state or self.States.PLAYING
#
#     def __cv_waitKey(self, delay=None):
#         if delay is None:
#             delay = self.delay
#         return cv2.waitKey(delay) & 0xFF
#
#     def __wait_pause_state(self):
#         if self.state != self.States.PAUSE:
#             raise Exception(f'state != {self.States.PAUSE}')
#
#         key = self.__cv_waitKey(-1)
#         if key == KbdKeys.Q:
#             self.state = self.States.PLAYING
#         return key
#
#     def __wait_normal_state(self):
#         if self.state != self.States.PLAYING:
#             raise Exception(f'state != {self.States.PLAYING}')
#
#         key = self.__cv_waitKey()
#         if key == KbdKeys.Q:
#             self.state = self.States.PAUSE
#             return self.__wait_pause_state()
#         return key
#
#     def waitKey(self):
#         if self.state == self.States.PAUSE:
#             return self.__wait_pause_state()
#         else:
#             return self.__wait_normal_state()
