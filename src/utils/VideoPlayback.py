import cv2
import math
from utils import leftClip
from utils.KbdKeys import KbdKeys
from utils.VideoPlaybackHandlerBase import VideoPlaybackHandlerBase
from utils.visualize import putFramePos


class VideoPlayback:
    def __init__(self, videoPath, initialFrameDelay=1, autoplayInitially=True):
        self.cap = cv2.VideoCapture(videoPath)
        self.frameDelay = initialFrameDelay
        self.autoPlay = autoplayInitially
        self._controller = VideoController(self)

    @property
    def manualPlay(self):
        return not self.autoPlay

    @manualPlay.setter
    def manualPlay(self, manualPlay):
        self.autoPlay = not manualPlay

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def frameSize(self):
        assert self.cap
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def __currentPos(self):
        assert self.cap
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def setPos(self, pos):
        assert self.cap
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    def __currentPosMsec(self):
        assert self.cap
        return self.cap.get(cv2.CAP_PROP_POS_MSEC)

    def backward(self, numFrames=1):
        assert self.cap
        newPos = self.__currentPos() - numFrames - 1
        if newPos < 0:
            newPos = 0
        self.setPos(newPos)

    def changeFrameDelay(self, direction):
        # manage change value depending of current frameDelay magnitude
        if self.frameDelay > 500:
            step = 20
        elif self.frameDelay > 100:
            step = 10
        elif self.frameDelay > 20:
            step = 5
        else:
            step = 1

        self.frameDelay += step * direction
        self.frameDelay = leftClip(self.frameDelay, 1)

    @staticmethod
    def __range(range):
        from_, to = 0, math.inf
        if not range:
            return from_, to
        if isinstance(range, int):
            from_ = range
            return from_, to
        if len(range) == 1:
            range = (range[0], math.inf)
        from_, to = range[:2]
        from_ = from_ or 0
        to = to or math.inf
        return from_, to

    def frames(self, range=None):
        assert self.cap
        from_, to = self.__range(range)
        if from_:
            self.setPos(from_)
        while True:
            pos = self.__currentPos()
            if pos > to:
                break
            posMsec = self.__currentPosMsec()
            frame = self.readFrame()
            if frame is None:
                break
            yield pos, frame, posMsec

    def readFrame(self, pos=None):
        if pos is not None:
            self.setPos(pos)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def framesCount(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def handleAction(self):
        return self._controller.handleAction()

    def play(self, range=None, onFrameReady=None, onStateChange=None):
        def defaultFrameReady(frame, framePos, framePosMsec, playback):
            cv2.imshow('Video', frame)

        def defaultStateChange(frameDelay, autoPlay, framePos, framePosMsec, playback):
            autoplayLabel = 'ON' if autoPlay else 'OFF'
            stateTitle = f'Video (Pos: {framePos}, FrameDelay: {frameDelay}, Autoplay: {autoplayLabel})'
            cv2.setWindowTitle('Video', stateTitle)

        def emptyStateChange(frameDelay, autoPlay, framePos, framePosMsec, playback):
            pass

        if onFrameReady is None and onStateChange is None:
            onStateChange = defaultStateChange
        if onStateChange is None:
            onStateChange = emptyStateChange
        onFrameReady = onFrameReady or defaultFrameReady

        for pos, frame, posMsec in self.frames(range):
            onFrameReady(frame, pos, posMsec, self)
            onStateChange(self.frameDelay, self.autoPlay, pos, posMsec, self)
            while True:
                read, stop, changed, key = self.handleAction()  # enter in user action handling
                if changed:
                    onStateChange(self.frameDelay, self.autoPlay, pos, posMsec, self)
                if read:
                    break
                if stop:
                    return False  # indicate interruption by user
        return True  # indicate end of video/frameSequence

    def playWithHandler(self, handler: VideoPlaybackHandlerBase, range=None):
        return self.play(range, handler.frameReady, handler.syncPlaybackState)


def readFrame(videoFile, framePos):
    pb = None
    try:
        pb = VideoPlayback(videoFile)
        return pb.readFrame(framePos)
    finally:
        pb and pb.release()


class VideoController:
    def __init__(self, videoPlayback):
        self.videoPlayback = videoPlayback
        self._scheduleManualPlay = False

    @staticmethod
    def waitKey(delay):
        return cv2.waitKeyEx(delay)

    BACKWARD_KEYS = {KbdKeys.L_ARROW_EX, KbdKeys.a, KbdKeys.A}
    FORWARD_KEYS = {KbdKeys.R_ARROW_EX, KbdKeys.d, KbdKeys.D}
    INCREASE_DELAY_KEYS = {KbdKeys.UP_ARROW_EX, KbdKeys.w, KbdKeys.W}
    DECREASE_DELAY_KEYS = {KbdKeys.DOWN_ARROW_EX, KbdKeys.s, KbdKeys.S}

    def _handleManualPlay(self):
        assert self.videoPlayback.manualPlay

        read, stop, changed = False, False, False
        key = self.waitKey(-1)

        if key == KbdKeys.Q:
            self.videoPlayback.autoPlay = True
            read, stop, changed = True, False, True

        elif key == KbdKeys.ESC:
            read, stop, changed = False, True, False

        elif key in self.BACKWARD_KEYS:
            self.videoPlayback.backward()
            read, stop, changed = True, False, False

        elif key in self.FORWARD_KEYS:
            read, stop, changed = True, False, False

        elif key in self.INCREASE_DELAY_KEYS:
            self.videoPlayback.changeFrameDelay(+1)
            read, stop, changed = False, False, True

        elif key in self.DECREASE_DELAY_KEYS:
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
            self.videoPlayback.manualPlay = True
            read, stop, changed = False, False, True

        elif key in self.BACKWARD_KEYS:
            self.videoPlayback.backward()
            self.videoPlayback.manualPlay = True
            read, stop, changed = True, False, True

        elif key in self.INCREASE_DELAY_KEYS:
            self.videoPlayback.changeFrameDelay(+1)
            read, stop, changed = False, False, True

        elif key in self.DECREASE_DELAY_KEYS:
            self.videoPlayback.changeFrameDelay(-1)
            read, stop, changed = False, False, True

        return read, stop, changed, key

    def handleAction(self):
        if self.videoPlayback.manualPlay:
            return self._handleManualPlay()
        else:
            return self._handleAutoPlay()


if __name__ == '__main__':
    def examples():
        def ex_1():
            def indicatePlaybackState(playback, winname):
                autoplayLabel = 'ON' if playback.autoPlay else 'OFF'
                stateTitle = f'Video (FrameDelay: {playback.frameDelay}, Autoplay: {autoplayLabel})'
                cv2.setWindowTitle(winname, stateTitle)

            def showFrameAndHandleActions(frame, playback):
                cv2.imshow('Video', frame)  # show frame
                indicatePlaybackState(playback, winname='Video')
                while True:
                    read, stop, changed, key = playback.handleAction()  # enter in user action handling
                    if changed:
                        indicatePlaybackState(playback, winname='Video')
                    if stop:
                        return True
                    if read:
                        return False

            # ---------------------------------------
            videoFile = '/HDD_DATA/Computer_Vision_Task/Video_2.mp4'
            videoPlayback = VideoPlayback(videoFile, 1000, autoplayInitially=False)

            for pos, frame in videoPlayback.frames():
                putFramePos(frame, pos)  # process frame
                stop = showFrameAndHandleActions(frame, videoPlayback)
                if stop:
                    break
            videoPlayback.release()

        def ex_2():
            videoFile = '/HDD_DATA/Computer_Vision_Task/Video_2.mp4'
            videoPlayback = VideoPlayback(videoFile, 1000, autoplayInitially=False)
            videoPlayback.play(onFrameReady=None, onStateChange=None)
            videoPlayback.release()

        def ex_3():
            winname = 'Video123'

            def indicatePlaybackState(frameDelay, autoPlay, framePos, playback):
                autoplayLabel = 'ON' if autoPlay else 'OFF'
                stateTitle = f'{winname} (FrameDelay: {frameDelay}, Autoplay: {autoplayLabel})'
                cv2.setWindowTitle(winname, stateTitle)

            def frameReady(frame, framePos, playback):
                putFramePos(frame, framePos)
                cv2.imshow(winname, frame)

            videoFile = '/HDD_DATA/Computer_Vision_Task/Video_2.mp4'
            videoPlayback = VideoPlayback(videoFile, 1000, autoplayInitially=False)
            videoPlayback.play(onFrameReady=frameReady, onStateChange=indicatePlaybackState)
            videoPlayback.release()
