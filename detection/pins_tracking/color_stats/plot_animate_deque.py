import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
from collections import deque

from utils import roundToInt
from utils.Timer import timeit


class images:
    @staticmethod
    def rndIntensities():
        from_, to = 0, 255
        current = np.random.randint(from_, to + 1)
        step = (np.random.random() + .5) * 1.5
        direction = 1 if np.random.randint(0, 2) == 0 else -1
        while True:
            yield roundToInt(current)
            current = current + step * direction
            if current <= from_ or current >= to:
                direction = -direction
                if current <= from_:
                    current = from_
                else:
                    current = to

    def __call__(self):
        channel = self.rndIntensities
        colors = zip(channel(), channel(), channel())
        for color in colors:
            img = np.empty([400, 500, 3], np.uint8)
            img[:, :] = color
            yield img


class FramePointColorPlotter:
    def __init__(self):
        self.point = None

        self.posData = deque(maxlen=300)
        self.colorData = deque(maxlen=300)

        fig, self.ax = plt.subplots()
        self.ax.set_ylim(0, 16777215)
        plt.show(block=False)

    def setPoint(self, point):
        self.point = point
        # clear data and plot
        self.posData.clear()
        self.colorData.clear()
        self.ax.clear()
        self.ax.set_ylim(0, 16777215)

    def drawPoint(self, img):
        if self.point is None:
            return
        x, y = self.point
        color = img[y, x]
        color = tuple(map(int, np.invert(color)))  # cant pass color as uint8 array...
        cv2.circle(img, self.point, 2, color=color, thickness=-1)

    @staticmethod
    def color24bit(img, point):
        x, y = point
        b, g, r = img[y, x]
        return int(b) + (int(g) << 8) + (int(r) << 16)

    def plotColor(self, pos, img):
        if self.point is None:
            return
        self.posData.append(pos)
        color24 = self.color24bit(img, self.point)
        self.colorData.append(color24)
        with timeit():
            self.ax.clear()
            self.ax.set_ylim(0, 16777215)
            self.ax.scatter(self.posData, self.colorData, s=1)
            # plt.pause(.01)
            plt.draw()

def main():
    def onClick(e, x, y, flags, _):
        if e != cv2.EVENT_LBUTTONDOWN:
            return
        plotter.setPoint((x, y))

    winname = 'Frame'
    cv2.namedWindow(winname)
    cv2.setMouseCallback(winname, onClick)

    plotter = FramePointColorPlotter()
    imgGen = images()
    for pos, img in enumerate(imgGen()):
        plotter.drawPoint(img)
        plotter.plotColor(pos, img)
        cv2.imshow(winname, img)
        # click: new point - new plot

        if cv2.waitKey(100) == 27:
            break


if __name__ == '__main__':
    main()
