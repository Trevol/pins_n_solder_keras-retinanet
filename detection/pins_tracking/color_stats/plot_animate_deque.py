import cv2

from detection.pins_tracking.color_stats.FramePointColorPlotter import FramePointColorPlotter
from utils.random_images import random_images


def main():
    def onClick(e, x, y, flags, _):
        if e != cv2.EVENT_LBUTTONDOWN:
            return
        plotter.setPoint((x, y))

    winname = 'Frame'
    cv2.namedWindow(winname)
    cv2.setMouseCallback(winname, onClick)

    plotter = FramePointColorPlotter()
    imgGen = random_images()
    for pos, img in enumerate(imgGen()):
        plotter.plotColor(pos, img)
        plotter.drawPoint(img)
        cv2.imshow(winname, img)
        # click: new point - new plot

        if cv2.waitKey(100) == 27:
            break


if __name__ == '__main__':
    main()
