import cv2
from detection.csv_cache.DetectionsCSV import DetectionsCSV
import utils.visualize
from utils.VideoPlayback import VideoPlayback
from detection.pins_tracking.v1.BBoxTracker import BBoxTracker


class VideoHandler:
    winname = 'Video'

    def __init__(self, framesDetections):
        self.framesDetections = framesDetections
        self.bboxTracker = BBoxTracker()

    def syncPlaybackState(self, frameDelay, autoPlay, framePos, playback):
        autoplayLabel = 'ON' if autoPlay else 'OFF'
        stateTitle = f'{self.winname} (FrameDelay: {frameDelay}, Autoplay: {autoplayLabel})'
        cv2.setWindowTitle(self.winname, stateTitle)

    def frameReady(self, frame, framePos, playback):
        detections = self.framesDetections.get(framePos, [])
        self.bboxTracker.applyRawFrameDetections(detections, framePos, frame)
        self.bboxTracker.draw(frame)
        utils.visualize.drawDetections(frame, detections)
        utils.visualize.putFramePos(frame, framePos)

        cv2.imshow(self.winname, frame)


def main():
    files = [
        ('/HDD_DATA/Computer_Vision_Task/Video_6.mp4',
         DetectionsCSV.readAsDict('../../csv_cache/data/detections_video6.csv'))
    ]

    for sourceVideoFile, framesDetections in files:
        videoPlayback = VideoPlayback(sourceVideoFile, 500, autoplayInitially=False)
        handler = VideoHandler(framesDetections)
        videoPlayback.play(range=(9, 184), onFrameReady=handler.frameReady, onStateChange=handler.syncPlaybackState)
        videoPlayback.release()
    cv2.waitKey()


if __name__ == '__main__':
    main()
