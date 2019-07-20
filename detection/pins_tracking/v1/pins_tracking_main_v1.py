import cv2
from detection.csv_cache.DetectionsCSV import DetectionsCSV
import utils.visualize
from utils.VideoPlayback import VideoPlayback
import numpy as np


class ROI:
    initialBbox = np.round([1501.1188, 785.05164, 1555.3042, 833.12695]).astype(np.int32)

    @classmethod
    def draw(cls, frame):
        x1, y1, x2, y2 = cls.initialBbox
        cv2.rectangle(frame, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (0, 255, 0), 2)


winname = 'Video'


def indicatePlaybackState(frameDelay, autoPlay, framePos, playback):
    autoplayLabel = 'ON' if autoPlay else 'OFF'
    stateTitle = f'{winname} (FrameDelay: {frameDelay}, Autoplay: {autoplayLabel})'
    cv2.setWindowTitle(winname, stateTitle)


def main():
    files = [
        ('/HDD_DATA/Computer_Vision_Task/Video_6.mp4',
         DetectionsCSV.readAsDict('../../csv_cache/data/detections_video6.csv'))
    ]

    def frameReady(frame, framePos, playback):
        detections = framesDetections.get(framePos, [])

        utils.visualize.drawDetections(frame, detections)
        # ROI.draw(frame)

        utils.visualize.putFramePos(frame, framePos)
        cv2.imshow(winname, frame)

    for sourceVideoFile, framesDetections in files:
        videoPlayback = VideoPlayback(sourceVideoFile, 500, autoplayInitially=False)
        videoPlayback.play(range=(9, 184), onFrameReady=frameReady, onStateChange=indicatePlaybackState)
        videoPlayback.release()
    cv2.waitKey()


if __name__ == '__main__':
    main()
