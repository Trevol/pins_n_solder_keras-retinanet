import numpy as np
import cv2


def onMouse(evt, x, y, flags, param):
    if evt == cv2.EVENT_LBUTTONDBLCLK:
        print(evt, x, y, flags)


def main():
    img = np.full([500, 400, 3], 150, np.uint8)
    cv2.imshow('img', img)
    cv2.setMouseCallback('img', onMouse)
    while cv2.waitKey() != 27:
        pass


main()
