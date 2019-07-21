import cv2
from detection.csv_cache.DetectionsCSV import DetectionsCSV
import utils.visualize
from utils import resize
from utils.VideoPlayback import VideoPlayback
from detection.pins_tracking.v1.InstanceTracker import InstanceTracker


class VideoHandler:
    winname = 'Video'

    def __init__(self, framesDetections):
        self.framesDetections = framesDetections
        self.instanceTracker = InstanceTracker()

    def syncPlaybackState(self, frameDelay, autoPlay, framePos, playback):
        autoplayLabel = 'ON' if autoPlay else 'OFF'
        stateTitle = f'{self.winname} (FrameDelay: {frameDelay}, Autoplay: {autoplayLabel})'
        cv2.setWindowTitle(self.winname, stateTitle)

    def frameReady(self, frame, framePos, playback):
        detections = [d for d in self.framesDetections.get(framePos, []) if d[-1] >= .9] #with score >= .9
        self.instanceTracker.applyRawFrameDetections(detections, framePos, frame)
        self.instanceTracker.draw(frame)
        utils.visualize.drawDetections(frame, detections)
        utils.visualize.putFramePos(frame, framePos)

        if frame.shape[1] >= 1900:  # fit view to screen
            frame = resize(frame, 0.7)
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
