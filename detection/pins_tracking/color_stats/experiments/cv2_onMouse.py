import numpy as np
import cv2


def onMouse(evt, x, y, flags, img):
    if evt == cv2.EVENT_LBUTTONDOWN:
        img[y, x] = 255
        cv2.imshow('img', img)


def main():
    img = np.full([500, 400, 3], 150, np.uint8)
    cv2.imshow('img', img)
    cv2.setMouseCallback('img', onMouse, img)
    while cv2.waitKey() != 27:
        pass


main()
