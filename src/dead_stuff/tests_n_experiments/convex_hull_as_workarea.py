import cv2

from utils.Colors import Colors
from utils import Geometry2D
from utils.Timer import timeit
from utils import readFrame
from detection.csv_cache.DetectionsCSV import DetectionsCSV
import utils


def frameInfo(framePos):
    video = '/HDD_DATA/Computer_Vision_Task/Video_6.mp4'
    frame = readFrame(video, framePos)
    detections = DetectionsCSV.loadPickle('../../../csv_cache/data/detections_video6.pcl')
    boxes = [d[0] for d in detections.get(framePos, []) if d[-1] >= .85]
    return frame, boxes


class App:
    winname = 'Frame'

    def __init__(self):
        self.frame, self.boxes = frameInfo(4150)
        self.hull = Geometry2D.convexHull(self.boxes)
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
