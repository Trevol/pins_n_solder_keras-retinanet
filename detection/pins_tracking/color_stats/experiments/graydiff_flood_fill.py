import cv2
import numpy as np

from utils.Timer import timeit


def putPoint(img, point):
    point2 = point[0] + 1, point[1] + 1
    cv2.rectangle(img, point, point2, 255, -1)


def getGrayDiff(baseThreshold, blur=True):
    grayDiff = cv2.imread('grayDiff_screenshot_30.08.2019.png', cv2.IMREAD_GRAYSCALE)
    if blur:
        grayDiff = cv2.blur(grayDiff, (3, 3))
    grayDiff = cv2.threshold(grayDiff, baseThreshold, 255, cv2.THRESH_TOZERO)[1]
    # cv2.imshow('trunc', grayDiff)
    return grayDiff


def floodFill(grayDiff, baseThreshold):
    # mask = np.zeros(shape=(grayDiff.shape[0] + 2, grayDiff.shape[1] + 2), dtype=np.uint8)
    mask = None
    seedPoint = 886, 77
    loDiff = baseThreshold
    upDiff = 255  # 87
    cv2.floodFill(grayDiff, mask, seedPoint, 255, loDiff, upDiff, 8)
    return grayDiff


def refine(filled):
    refined = cv2.threshold(filled, 254, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.fillPoly(refined, contours, 255)
    return refined


def main():
    baseThreshold = 10

    with timeit(autoreport=True):
        for _ in range(100):
            grayDiff = getGrayDiff(baseThreshold)
            filled = floodFill(grayDiff.copy(), baseThreshold)
            refined = refine(filled)

    # cv2.imshow('grayDiff', grayDiff)
    # cv2.imshow('filled', filled)
    # cv2.imshow('refined', refined)
    # cv2.waitKeyEx()


main()
