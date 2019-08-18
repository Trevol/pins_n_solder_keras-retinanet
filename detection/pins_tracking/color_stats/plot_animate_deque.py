import numpy as np
import cv2

from detection.pins_tracking.color_stats.FramePointColorPlotter import FramePointColorPlotter
from utils import roundToInt


class images:
    @staticmethod
    def rndIntensities():
        def next_(current, direction, step):
            current = current + step * direction
            if current <= from_ or current >= to:
                direction = -direction
                if current <= from_:
                    current = from_
                else:
                    current = to
            return current, direction

        from_, to = 0, 255
        current = np.random.randint(from_, to + 1)
        step = (np.random.random() + .5) * 1.5
        direction = 1 if np.random.randint(0, 2) == 0 else -1
        while True:
            yield roundToInt(current)
            current, direction = next_(current, direction, step)

    def __call__(self):
        channel = self.rndIntensities
        colors = zip(channel(), channel(), channel())
        for color in colors:
            img = np.empty([400, 500, 3], np.uint8)
            img[:, :] = color
            yield img


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
