import numpy as np
import cv2
import math

from detection.pins_tracking.v1.Colors import Colors
from utils.Timer import timeit
from utils.VideoPlayback import readFrame
from detection.csv_cache.DetectionsCSV import DetectionsCSV
import utils.visualize


def frameInfo(framePos):
    video = '/HDD_DATA/Computer_Vision_Task/Video_6.mp4'
    frame = readFrame(video, framePos)
    detections = DetectionsCSV.loadPickle('../../../csv_cache/data/detections_video6.pcl')
    boxes = [d[0] for d in detections.get(framePos, []) if d[-1] >= .85]
    return frame, boxes


class Geo2D:
    @staticmethod
    def boxPoints(box):
        x0 = math.floor(box[0])
        y0 = math.floor(box[1])
        x1 = math.ceil(box[2])
        y1 = math.ceil(box[3])

        topLeft = x0, y0
        yield topLeft
        bottomRight = x1, y1
        yield bottomRight
        bottomLeft = x0, y1
        yield bottomLeft
        topRight = x1, y0
        yield topRight

    @staticmethod
    def convexHull(boxes):
        pts = np.int32([pt for b in boxes for pt in Geo2D.boxPoints(b)])
        return cv2.convexHull(pts)


class App:
    winname = 'Frame'

    def __init__(self):
        self.frame, self.boxes = frameInfo(4150)
        self.hull = Geo2D.convexHull(self.boxes)
        self.__draw_state()
        cv2.namedWindow(self.winname)
        cv2.setMouseCallback(self.winname, self.onMouseEvent)

    def __draw_state(self):
        utils.visualize.drawBoxes(self.frame, self.boxes)
        cv2.drawContours(self.frame, [self.hull], 0, Colors.green, 2)

    def onMouseEvent(self, evt, x, y, flags, param):
        if evt != cv2.EVENT_LBUTTONUP:
            return
        with timeit():
            dist = cv2.pointPolygonTest(self.hull, (x, y), True)
        print('dist', dist)

    def show(self):
        cv2.imshow(self.winname, self.frame)
        cv2.waitKey()


if __name__ == '__main__':
    App().show()
