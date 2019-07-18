from sandbox.VideoPlayback2 import VideoPlayback2, VideoPlaybackActionResult
from detection.visualize import putFramePos
import cv2
from utils.KbdKeys import KbdKeys


def main():
    videoFile = '/HDD_DATA/Computer_Vision_Task/Video_2.mp4'

    videoPlayback = VideoPlayback2(videoFile, 1000, autoplayInitially=True)

    for pos, frame in videoPlayback.frames():

        putFramePos(frame, pos)  # process frame

        # break playback
        # or read/play next frame
        # or change speed
        # or on/off autoplay

        while True:
            cv2.imshow('Video', frame)  # show frame

            action, key = videoPlayback.handleAction()  # enter to user action handling
            if action in (VideoPlaybackActionResult.NextFrame, VideoPlaybackActionResult.Break):
                break
            if action == VideoPlaybackActionResult.ChangeState:
                stateTitle = f'Video (FrameDelay: {videoPlayback.frameDelay}, Autoplay: {videoPlayback.autoPlay})'
                cv2.setWindowTitle('Video', stateTitle)
        if action == VideoPlaybackActionResult.Break:
            break

    videoPlayback.release()


if __name__ == '__main__':
    main()
