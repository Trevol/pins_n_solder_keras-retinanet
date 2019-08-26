import cv2
import numpy as np

from utils.VideoPlayback import VideoPlayback


def onMouse(evt, x, y, flags, param):
    if evt != cv2.EVENT_LBUTTONDOWN:
        return
    img, imgHsv = param
    h, s, v = imgHsv[y, x]
    sensivity = 15
    imgH = imgHsv[..., 0]
    # mask = np.bitwise_not((imgH > (h - sensivity)) | (imgH < (h + sensivity)))
    mask = (imgH < (h + sensivity)) & (imgH > (h - sensivity))
    img = img.copy()
    img[mask] = 0
    cv2.imshow('Result', img)


def main():
    frame = cv2.imread('6_938.png')
    frameHsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('Frame', frame)
    cv2.setMouseCallback('Frame', onMouse, (frame, frameHsv))
    cv2.imshow('Hue', frameHsv[..., 0])

    while cv2.waitKey() != 27: pass


main()
