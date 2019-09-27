import numpy as np
import cv2


def main():
    f7 = cv2.imread('7.png', cv2.IMREAD_GRAYSCALE)
    f13 = cv2.imread('13.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('diff1', cv2.subtract(f13, f7))
    cv2.imshow('diff1', cv2.subtract(f7, f13))
    cv2.waitKey()


main()
