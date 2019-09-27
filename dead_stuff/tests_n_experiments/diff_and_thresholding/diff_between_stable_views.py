import cv2
import numpy as np

from utils import resize
from utils.Timer import timeit
from utils.VideoPlayback import VideoPlayback


def getConsequentFrames():
    frames = 3667, 3716
    # frames = 2436, 2633
    # frames = (420, 586)
    p = VideoPlayback('/HDD_DATA/Computer_Vision_Task/Video_6.mp4', 1, autoplayInitially=False)
    f1 = p.readFrame(frames[0])
    f2 = p.readFrame(frames[1])
    return f1, f2


def makeDiff(f1, f2):
    grayF1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    grayF2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    # TODO: compute gradients and try extract peaks/ridges(хребты). By ridgeDetector?

    # наши "эллипсы" самые светлые - свет падает на них и хорошо отражается в камеру
    # cv2.subtract отриц. значения (более темные) устанавливает в 0, что и требуется
    grayDiff = cv2.subtract(grayF2, grayF1)

    return grayDiff


def getMaskByConstantThreshold(grayDiff, threshold=10):
    return cv2.threshold(grayDiff, 10, 255, cv2.THRESH_BINARY)


def getMaskByUpdatedOtsu(grayDiff):
    otsuThr = cv2.threshold(grayDiff, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
    return cv2.threshold(grayDiff, otsuThr - 30, 255, cv2.THRESH_BINARY)


def getMaskByMedian(grayDiff):
    thr = np.median(grayDiff)
    return cv2.threshold(grayDiff, thr, 255, cv2.THRESH_BINARY)


def getMaskByMean(grayDiff):
    thr = np.mean(grayDiff)
    return cv2.threshold(grayDiff, thr, 255, cv2.THRESH_BINARY)


def getMaskByMeanEnhanced(grayDiff):
    # resize
    # exclude 0
    # calc mean
    small = np.float32(resize(grayDiff, 4))
    small[small == 0] = np.nan
    thr = np.nanmean(small)

    return cv2.threshold(grayDiff, thr, 255, cv2.THRESH_BINARY)


def showBlobsOnDiff():
    def s(f):
        # return f
        return f[100:500, 350:1600]

    f1, f2 = getConsequentFrames()
    grayDiff = makeDiff(f1, f2)
    constantThreshold, maskByConstantThreshold = getMaskByConstantThreshold(grayDiff)
    updatedOtsuThreshold, maskByUpdatedOtsu = getMaskByUpdatedOtsu(grayDiff)
    median, maskByMedian = getMaskByMedian(grayDiff)
    print(median)
    mean, maskByMean = getMaskByMean(grayDiff)
    print(mean)

    meanEnh, maskByMeanEnhanced = getMaskByMeanEnhanced(grayDiff)
    print(meanEnh)

    # show results
    cv2.imshow('grayDiff', s(grayDiff))
    cv2.imshow('maskByConstantThreshold', s(maskByConstantThreshold))
    cv2.imshow('maskByUpdatedOtsu', s(maskByUpdatedOtsu))
    cv2.imshow('maskByMedian', s(maskByMedian))
    cv2.imshow('maskByMean', s(maskByMean))
    cv2.imshow('maskByMeanEnhanced', s(maskByMeanEnhanced))

    while cv2.waitKey() != 27: pass


def main():
    # TODO: detect blobs on mask and grayDiff(with thresholding)
    showBlobsOnDiff()
    # playVideo()


main()
