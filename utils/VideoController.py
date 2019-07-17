import cv2
from .KbdKeys import KbdKeys


class VideoController:
    def __init__(self, delay, state=None):
        self.delay = delay
        self.state = state or 'normal'

    def __cv_waitKey(self, delay=None):
        if delay is None:
            delay = self.delay
        return cv2.waitKey(delay) & 0xFF

    def __wait_pause_state(self):
        if self.state != 'pause':
            raise Exception('state != pause')
        key = self.__cv_waitKey(-1)
        if key == KbdKeys.Q:
            self.state = 'normal'
        return key

    def __wait_normal_state(self):
        if self.state != 'normal':
            raise Exception('state != normal')

        key = self.__cv_waitKey()
        if key == KbdKeys.Q:
            self.state = 'pause'
            return self.__wait_pause_state()
        return key

    def waitKey(self):
        if self.state == 'pause':
            return self.__wait_pause_state()
        else:
            return self.__wait_normal_state()
