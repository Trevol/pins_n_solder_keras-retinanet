import cv2
import numpy as np

from utils import resize
from utils.Timer import timeit
from utils.VideoPlayback import VideoPlayback


def getConsequentFrames():
    frames = 2, 3716  # emptyBackground and pinsArray
    # frames = 8292, 8361 # solder added
    # frames = 3667, 3716
    # frames = 2436, 2633
    # frames = (420, 586)
    p = VideoPlayback('/HDD_DATA/Computer_Vision_Task/Video_6.mp4', 1, autoplayInitially=False)
    f1 = p.readFrame(frames[0])
    f2 = p.readFrame(frames[1])
    return f1, f2


def makeDiff(f1, f2):
    # TODO: compute gradients and try extract peaks/ridges(хребты). By ridgeDetector?

    # наши "эллипсы" самые светлые - свет падает на них и хорошо отражается в камеру
    # cv2.subtract отриц. значения (более темные) устанавливает в 0, что и требуется
    colorDiff = cv2.subtract(f1, f2)

    return colorDiff


def toGrayscale(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def showBlobsOnDiff():
    def s(f):
        # return f
        return f[100:-100, 350:1600]

    f1, f2 = getConsequentFrames()
    diff = cv2.subtract(f1, f2)
    # diff = cv2.absdiff(f1, f2)
    grayF1 = toGrayscale(f1)
    grayF2 = toGrayscale(f2)
    diffOfGrays = cv2.subtract(grayF1, grayF2)
    # show results
    cv2.imshow('f1', s(f1))
    cv2.imshow('f2', s(f2))
    cv2.imshow('diff', s(diff))
    cv2.imshow('grayDiff', s(toGrayscale(diff)))
    cv2.imshow('diffOfGrays', s(diffOfGrays))

    while cv2.waitKey() != 27: pass


def main():
    # TODO: detect blobs on mask and grayDiff(with thresholding)
    showBlobsOnDiff()
    # playVideo()


main()
