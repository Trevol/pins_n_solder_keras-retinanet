from sandbox.VideoPlayback2 import VideoPlayback2
from detection.visualize import putFramePos
import cv2


def main():
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
    videoPlayback = VideoPlayback2(videoFile, 1000, autoplayInitially=False)

    for pos, frame in videoPlayback.frames():
        putFramePos(frame, pos)  # process frame
        stop = showFrameAndHandleActions(frame, videoPlayback)
        if stop:
            break
    videoPlayback.release()


def proto():
    pass


if __name__ == '__main__':
    main()
