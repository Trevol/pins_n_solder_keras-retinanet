import cv2
from detection.csv_cache.DetectionsCSV import DetectionsCSV
import utils.visualize
from utils import resize
from utils.VideoPlayback import VideoPlayback
from detection.pins_tracking.v1.TechProcessTracker import TechProcessTracker


class VideoHandler:
    winname = 'Video'

    def __init__(self, framesDetections, workBox):
        self.framesDetections = framesDetections
        self.workBox = workBox
        self.techProcessTracker = TechProcessTracker()

    def syncPlaybackState(self, frameDelay, autoPlay, framePos, framePosMsec, playback):
        autoplayLabel = 'ON' if autoPlay else 'OFF'
        stateTitle = f'{self.winname} (FrameDelay: {frameDelay}, Autoplay: {autoplayLabel})'
        cv2.setWindowTitle(self.winname, stateTitle)

    def frameReady(self, frame, framePos, framePosMsec, playback):
        frameDetections = [d for d in self.framesDetections.get(framePos, []) if
                           d[-1] >= .85 and self.inWorkBox(d[0], self.workBox)]  # with score >= someThresh

        self.techProcessTracker.track(frameDetections, framePos, framePosMsec, frame)
        self.techProcessTracker.draw(frame)

        utils.visualize.drawDetections(frame, frameDetections)
        utils.visualize.putFramePos(frame, framePos, framePosMsec)

        if frame.shape[1] >= 1900:  # fit view to screen
            frame = resize(frame, 0.7)
        cv2.imshow(self.winname, frame)

    @staticmethod
    def inWorkBox(box, workBox):
        wx0, wy0, wx1, wy1 = workBox
        x0, y0, x1, y1 = box
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        # box center in workBox
        return wx0 < cx < wx1 and wy0 < cy < wy1


def files():
    yield ('/HDD_DATA/Computer_Vision_Task/Video_6.mp4',
           DetectionsCSV.loadPickle('../../csv_cache/data/detections_video6.pcl'),
           (222 // 0.7, 70 // 0.7, 1162 // 0.7, 690 // 0.7))
    yield ('/HDD_DATA/Computer_Vision_Task/Video_2.mp4',
           DetectionsCSV.loadPickle('../../csv_cache/data/detections_video2.pcl'),
           (147, 87, 1005, 669))


def main():
    for sourceVideoFile, framesDetections, workBox in files():
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=False)
        handler = VideoHandler(framesDetections, workBox)

        framesRange = (0, None)
        # framesRange = None
        videoPlayback.play(range=framesRange, onFrameReady=handler.frameReady, onStateChange=handler.syncPlaybackState)
        videoPlayback.release()

    cv2.waitKey()


if __name__ == '__main__':
    main()
