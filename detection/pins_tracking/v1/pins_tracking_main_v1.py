import cv2
from detection.csv_cache.DetectionsCSV import DetectionsCSV
import utils.visualize
from utils import resize
from utils.VideoPlayback import VideoPlayback
from utils import videoWriter

from detection.pins_tracking.v1.TechProcessTracker import TechProcessTracker
from detection.pins_tracking.v1.VideoConfig import video6SolderConfig


class VideoHandler:
    winname = 'Video'

    def __init__(self, framesDetections, sldConfig, videoWriter):
        self.videoWriter = videoWriter
        self.framesDetections = framesDetections
        self.techProcessTracker = TechProcessTracker(sldConfig)
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
        frameDetections = self.framesDetections.get(framePos, [])
        self.techProcessTracker.track(frameDetections, framePos, framePosMsec, frame)

        self.techProcessTracker.draw(frame)

        utils.visualize.drawDetections(frame, frameDetections)
        utils.visualize.putFramePos(frame, framePos, framePosMsec)
        self.techProcessTracker.drawStats(frame)

        imshowFrame = frame
        if imshowFrame.shape[1] >= 1900:  # fit view to screen
            imshowFrame = resize(frame, 0.7)
        cv2.imshow(self.winname, imshowFrame)
        self.videoWriter and self.videoWriter.write(frame)


def files():
    workArea6 = None  # (222 // 0.7, 70 // 0.7, 1162 // 0.7, 690 // 0.7)
    workArea2 = None  # (147, 87, 1005, 669)
    yield ('/HDD_DATA/Computer_Vision_Task/Video_6.mp4',
           '/HDD_DATA/Computer_Vision_Task/Video_6_result.mp4',
           DetectionsCSV.loadPickle('../../csv_cache/data/detections_video6.pcl'),
           workArea6,
           video6SolderConfig)

    # yield ('/HDD_DATA/Computer_Vision_Task/Video_2.mp4',
    #        '/HDD_DATA/Computer_Vision_Task/Video_2_result.mp4',
    #        DetectionsCSV.loadPickle('../../csv_cache/data/detections_video2.pcl'),
    #        workArea2, None)


def main():
    for sourceVideoFile, resultVideo, framesDetections, _, cfg in files():
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=False)
        videoWriter = None  # videoWriter(videoPlayback.cap, resultVideo)
        handler = VideoHandler(framesDetections, cfg, videoWriter)

        # framesRange = (4150, None)
        # framesRange = (8100, None)
        framesRange = None
        videoPlayback.play(range=framesRange, onFrameReady=handler.frameReady, onStateChange=handler.syncPlaybackState)
        videoPlayback.release()
        cv2.waitKey()
        handler.release()
        videoWriter and videoWriter.release()

    cv2.waitKey()


if __name__ == '__main__':
    main()
