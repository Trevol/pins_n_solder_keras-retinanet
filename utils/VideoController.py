import cv2
from .KbdKeys import KbdKeys


class VideoController:
    class States:
        PAUSE = 'pause'
        PLAYING = 'playing'

    def __init__(self, videoPlayback, initialFrameDelay=1, initialyPaused=False):
        self.videoPlayback = videoPlayback
        self.frameDelay = initialFrameDelay
        self.state = self.States.PAUSE if initialyPaused else self.States.PLAYING

    def __cv_waitKey(self, delay=None):
        if delay is None:
            delay = self.delay
        return cv2.waitKey(delay) & 0xFF

    def __wait_pause_state(self):
        if self.state != self.States.PAUSE:
            raise Exception(f'state != {self.States.PAUSE}')

        key = self.__cv_waitKey(-1)
        if key == KbdKeys.Q:
            self.state = self.States.PLAYING
        return key

    def __wait_normal_state(self):
        if self.state != self.States.PLAYING:
            raise Exception(f'state != {self.States.PLAYING}')

        key = self.__cv_waitKey()
        if key == KbdKeys.Q:
            self.state = self.States.PAUSE
            return self.__wait_pause_state()
        return key

    def handleKey(self):
        if self.state == self.States.PAUSE:
            return self.__wait_pause_state()
        else:
            return self.__wait_normal_state()

    def handleKey___(self):
        key = cv2.waitKey()
        if key == KbdKeys.L_ARROW:
            self.videoPlayback.backward()
        return key

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
