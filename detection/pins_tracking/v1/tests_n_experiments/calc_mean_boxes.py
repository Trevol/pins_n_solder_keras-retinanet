import cv2
from detection.csv_cache.DetectionsCSV import DetectionsCSV
import utils.visualize
from detection.pins_tracking.v1.Box import Box
from utils import resize
from utils.VideoPlayback import VideoPlayback


class BoxStats:
    def __init__(self):
        self.framesBBoxes = []
        self.meanBBoxes = []

    def update(self, detections):
        bboxes = [Box(d[0]) for d in detections]

        if not any(self.framesBBoxes):
            self.framesBBoxes.append(bboxes)
            self.meanBBoxes = bboxes
            return

        assert len(bboxes) == len(self.framesBBoxes[0])
        bboxes = self.orderByInstance(bboxes)
        self.framesBBoxes.append(bboxes)
        self.calcStats()

    def calcStats(self):
        # calc meanBBoxes
        countOfMeanBoxes = len(self.framesBBoxes[0])
        for instanceIndex in range(countOfMeanBoxes):
            instanceBoxesAcrossFrames = [frameBoxes[instanceIndex] for frameBoxes in self.framesBBoxes]
            instanceMeanBox = Box.meanBox(instanceBoxesAcrossFrames)
            self.meanBBoxes[instanceIndex] = instanceMeanBox

        # calc std
        # raise NotImplementedError()

    def orderByInstance(self, boxes):
        # reorder by instanceIndex (defined in frameBoxes)
        # index, box, dist = meanBb.nearest(boxes)
        return [meanBb.nearest(boxes)[1] for meanBb in self.meanBBoxes]

    def drawMeanBoxes(self, image):
        green = (0, 200, 0)
        for b in self.meanBBoxes:
            cv2.rectangle(image, tuple(b.pt0), tuple(b.pt1), green)
            cv2.circle(image, tuple(b.center), 1, green, -1)


class VideoHandler:
    winname = 'Video'

    def __init__(self, framesDetections):
        self.framesDetections = framesDetections
        # self.techProcessTracker = TechProcessTracker()
        self.boxStats = BoxStats()

    def syncPlaybackState(self, frameDelay, autoPlay, framePos, framePosMsec, playback):
        autoplayLabel = 'ON' if autoPlay else 'OFF'
        stateTitle = f'{self.winname} (FrameDelay: {frameDelay}, Autoplay: {autoplayLabel})'
        cv2.setWindowTitle(self.winname, stateTitle)

    def frameReady(self, frame, framePos, framePosMsec, playback):
        frameDetections = [d for d in self.framesDetections.get(framePos, []) if d[-1] >= .9]  # with score >= .9

        # print??
        self.boxStats.update(frameDetections)
        meanOutput = frame.copy()
        self.boxStats.drawMeanBoxes(meanOutput)
        if meanOutput.shape[1] >= 1900:  # fit view to screen
            meanOutput = resize(meanOutput, 0.7)
        cv2.imshow('Mean', meanOutput)

        utils.visualize.drawDetections(frame, frameDetections, drawCenters=True)
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
        videoPlayback.play(range=(56, 133), onFrameReady=handler.frameReady, onStateChange=handler.syncPlaybackState)
        videoPlayback.release()
    cv2.waitKey()


if __name__ == '__main__':
    main()