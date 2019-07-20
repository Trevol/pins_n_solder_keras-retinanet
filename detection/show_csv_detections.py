import cv2
from detection.DetectionsCSV import DetectionsCSV
import utils.visualize


def main():
    from sandbox.VideoPlayback2 import VideoPlayback2
    winname = 'Video'

    def indicatePlaybackState(frameDelay, autoPlay, framePos, playback):
        autoplayLabel = 'ON' if autoPlay else 'OFF'
        stateTitle = f'{winname} (FrameDelay: {frameDelay}, Autoplay: {autoplayLabel})'
        cv2.setWindowTitle(winname, stateTitle)

    files = [
        ('/HDD_DATA/Computer_Vision_Task/Video_6.mp4',
         DetectionsCSV.readAsDict('./data/detections_video6.csv')),

        ('/HDD_DATA/Computer_Vision_Task/Video_2.mp4',
         DetectionsCSV.readAsDict('./data/detections_video2.csv'))
    ]

    for sourceVideoFile, framesDetections in files:
        def frameReady(frame, framePos, playback):
            detections = framesDetections.get(framePos, [])
            utils.visualize.drawDetections(frame, detections)
            utils.visualize.putFramePos(frame, framePos)
            cv2.imshow(winname, frame)

        videoPlayback = VideoPlayback2(sourceVideoFile, 500, autoplayInitially=False)
        videoPlayback.play(onFrameReady=frameReady, onStateChange=indicatePlaybackState)
        videoPlayback.release()


if __name__ == '__main__':
    main()
