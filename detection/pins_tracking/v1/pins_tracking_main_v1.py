import cv2
from detection.csv_cache.DetectionsCSV import DetectionsCSV
import utils.visualize
from utils import resize
from utils.VideoPlayback import VideoPlayback
from detection.pins_tracking.v1.TechProcessTracker import TechProcessTracker


class VideoHandler:
    winname = 'Video'

    def __init__(self, framesDetections):
        self.framesDetections = framesDetections
        self.techProcessTracker = TechProcessTracker()

    def syncPlaybackState(self, frameDelay, autoPlay, framePos, framePosMsec, playback):
        autoplayLabel = 'ON' if autoPlay else 'OFF'
        stateTitle = f'{self.winname} (FrameDelay: {frameDelay}, Autoplay: {autoplayLabel})'
        cv2.setWindowTitle(self.winname, stateTitle)

    def frameReady(self, frame, framePos, framePosMsec, playback):
        frameDetections = [d for d in self.framesDetections.get(framePos, []) if d[-1] >= .9]  # with score >= .9

        self.techProcessTracker.track(frameDetections, framePos, framePosMsec, frame)
        self.techProcessTracker.draw(frame)

        utils.visualize.drawDetections(frame, frameDetections)
        utils.visualize.putFramePos(frame, framePos, framePosMsec)

        if frame.shape[1] >= 1900:  # fit view to screen
            frame = resize(frame, 0.7)
        cv2.imshow(self.winname, frame)


def files():
    yield ('/HDD_DATA/Computer_Vision_Task/Video_6.mp4',
           DetectionsCSV.loadPickle('../../csv_cache/data/detections_video6.pcl'))


def main():
    for sourceVideoFile, framesDetections in files():
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=False)
        handler = VideoHandler(framesDetections)
        # framesRange = (9, 184)
        framesRange = None
        videoPlayback.play(range=framesRange, onFrameReady=handler.frameReady, onStateChange=handler.syncPlaybackState)
        videoPlayback.release()
    cv2.waitKey()


if __name__ == '__main__':
    main()
