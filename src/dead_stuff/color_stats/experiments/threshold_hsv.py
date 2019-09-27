import cv2
import numpy as np
from itertools import cycle

from utils import VideoPlayback


def threshold__(imgHsv, x, y):
    sensitivity = 15
    h = imgHsv[y, x, 0]
    lower_red_0 = np.array([0, 100, 100])
    upper_red_0 = np.array([sensitivity, 255, 255])
    lower_red_1 = np.array([180 - sensitivity, 100, 100])
    upper_red_1 = np.array([180, 255, 255])

    mask_0 = cv2.inRange(imgHsv, lower_red_0, upper_red_0);
    mask_1 = cv2.inRange(imgHsv, lower_red_1, upper_red_1);
    mask = cv2.bitwise_or(mask_0, mask_1)
    return mask.astype(bool)


def threshold(imgHsv, x, y):
    sensitivity = 3
    h = imgHsv[y, x, 0]
    s = imgHsv[y, x, 1]
    v = imgHsv[y, x, 2]

    lower = np.array([h - sensitivity, s - 20, v - 50])
    upper = np.array([h + sensitivity, 255, 255])

    mask = cv2.inRange(imgHsv, lower, upper);

    return mask.astype(bool)


counter09 = cycle(range(10))


def onMouse(evt, x, y, flags, param):
    if evt != cv2.EVENT_LBUTTONDOWN:
        return
    img, imgHsv = param
    print(next(counter09), 'hsv:', imgHsv[y, x], 'bgr:', img[y, x])
    if flags & cv2.EVENT_FLAG_CTRLKEY:
        if np.array_equal(img[y, x], [0, 0, 0]):
            return
        mask = threshold(imgHsv, x, y)
        img[mask] = 0
        cv2.imshow('Frame', img)

def getFrame():
    return VideoPlayback('/HDD_DATA/Computer_Vision_Task/Video_6.mp4').readFrame(3938)
    return cv2.imread('6_938.png')

def main():
    frame = getFrame()
    frameHsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('Frame', frame)
    cv2.setMouseCallback('Frame', onMouse, (frame, frameHsv))

    while cv2.waitKey() != 27: pass


main()
