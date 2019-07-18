import cv2
from .KbdKeys import KbdKeys
from . import leftClip


class VideoController:
    class States:
        PAUSE = 'paused'
        PLAYING = 'playing'

    def __init__(self, videoPlayback, initialFrameDelay=1, initialyPaused=False):
        self.videoPlayback = videoPlayback
        self.frameDelay = initialFrameDelay
        self.state = self.States.PAUSE if initialyPaused else self.States.PLAYING
        self._schedulePause = False

    def _enterPausedState(self):
        assert self.state != self.States.PAUSE
        self._schedulePause = False
        self.state = self.States.PAUSE
        return self.__inPausedState()

    def __inPausedState(self):
        assert self.state == self.States.PAUSE

        key = cv2.waitKey(-1)

        if key == KbdKeys.Q:
            self._enterPlayingState()
        elif key == KbdKeys.L_ARROW:
            self.videoPlayback.backward()
        elif key == KbdKeys.UP_ARROW:
            self._speedUp()
        elif key == KbdKeys.DOWN_ARROW:
            self._speedDown()

        return key

    def _speedUp(self):
        self.frameDelay -= 1  # TODO: manage step depending of frameDelay magnitude
        self.frameDelay = leftClip(self.frameDelay, 1)

    def _speedDown(self):
        self.frameDelay -= 1  # TODO: manage step depending of frameDelay magnitude
        self.frameDelay = leftClip(self.frameDelay, 1)

    def _enterPlayingState(self):
        assert self.state != self.States.PLAYING
        self.state = self.States.PLAYING

    def __inPlayingState(self):
        assert self.state == self.States.PLAYING
        key = cv2.waitKey(self.frameDelay)

        if key == KbdKeys.Q:
            return self._enterPausedState()
        elif key == KbdKeys.L_ARROW:
            self.videoPlayback.backward()
            self._schedulePause = True  # return outside for frame retrieval and then enterPause
        elif key == KbdKeys.UP_ARROW:
            self._speedUp()
        elif key == KbdKeys.DOWN_ARROW:
            self._speedDown()

        return key

    def handleKey(self):
        if self._schedulePause:
            return self._enterPausedState()
        if self.state == self.States.PAUSE:
            return self.__inPausedState()
        else:
            return self.__inPlayingState()

    # def handleKey___(self):
    #     key = cv2.waitKey()
    #     if key == KbdKeys.L_ARROW:
    #         self.videoPlayback.backward()
    #     return key

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
