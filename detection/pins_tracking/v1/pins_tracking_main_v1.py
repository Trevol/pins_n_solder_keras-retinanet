import cv2
from detection.csv_cache.DetectionsCSV import DetectionsCSV
import utils.visualize
from utils import resize
from utils.VideoPlayback import VideoPlayback
from utils import videoWriter

from detection.pins_tracking.v1.TechProcessTracker import TechProcessTracker
from detection.pins_tracking.v1.VideoConfig import config


class VideoHandler:
    winname = 'Video'

    def __init__(self, framesDetections, workBox, cfg, writer):
        self.writer = writer
        self.framesDetections = framesDetections
        self.workBox = workBox
        self.techProcessTracker = TechProcessTracker(cfg)
        cv2.namedWindow(self.winname)
        cv2.setMouseCallback(self.winname, self.onMouse)

    def release(self):
        cv2.destroyWindow(self.winname)

    def onMouse(self, evt, x, y, flags, param):
        if evt != cv2.EVENT_LBUTTONUP:
            return
        pt = (int(round(x / .7)), int(round(y / .7)))
        self.techProcessTracker.dumpPinStats(pt)

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
        self.techProcessTracker.drawStats(frame)

        imshowFrame = frame
        if imshowFrame.shape[1] >= 1900:  # fit view to screen
            imshowFrame = resize(frame, 0.7)
        cv2.imshow(self.winname, imshowFrame)
        self.writer and self.writer.write(frame)

    @staticmethod
    def inWorkBox(box, workBox):
        wx0, wy0, wx1, wy1 = workBox
        x0, y0, x1, y1 = box
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        # box center in workBox
        return wx0 < cx < wx1 and wy0 < cy < wy1


def files():
    # TODO: detect workBox automatically
    yield ('/HDD_DATA/Computer_Vision_Task/Video_6.mp4',
           '/HDD_DATA/Computer_Vision_Task/Video_6_result.mp4',
           DetectionsCSV.loadPickle('../../csv_cache/data/detections_video6.pcl'),
           (222 // 0.7, 70 // 0.7, 1162 // 0.7, 690 // 0.7),
           config)

    yield ('/HDD_DATA/Computer_Vision_Task/Video_2.mp4',
           '/HDD_DATA/Computer_Vision_Task/Video_2_result.mp4',
           DetectionsCSV.loadPickle('../../csv_cache/data/detections_video2.pcl'),
           (147, 87, 1005, 669), None)


def main():
    for sourceVideoFile, resultVideo, framesDetections, workBox, cfg in files():
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=False)
        writer = None  # videoWriter(videoPlayback.cap, resultVideo)
        handler = VideoHandler(framesDetections, workBox, cfg, writer)

        # framesRange = (4150, None)
        # framesRange = (8100, None)
        framesRange = None
        videoPlayback.play(range=framesRange, onFrameReady=handler.frameReady, onStateChange=handler.syncPlaybackState)
        videoPlayback.release()
        cv2.waitKey()
        handler.release()
        writer and writer.release()

    cv2.waitKey()


if __name__ == '__main__':
    main()
