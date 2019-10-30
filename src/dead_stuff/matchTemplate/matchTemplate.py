import numpy as np
import cv2

from utils.Timer import timeit


def main():
    image = np.random.randint(0, 256, [800, 1300], np.uint8)
    image[500: 550, 1000:1050] = 127
    tpl = np.full([50, 50], 127, np.uint8)

    for _ in range(3):
        with timeit():
            r1 = cv2.matchTemplate(image, tpl, cv2.TM_SQDIFF_NORMED)
    print('------------------')
    for _ in range(3):
        with timeit():
            r2 = cv2.matchTemplate(image, tpl, cv2.TM_CCORR_NORMED)
    print('------------------')
    for _ in range(3):
        with timeit():
            r1 = cv2.matchTemplate(image, tpl, cv2.TM_SQDIFF_NORMED)

    return
    cv2.imshow('image', image)
    cv2.imshow('r1', np.uint8(r1*255))
    cv2.imshow('r2', np.uint8(r2*255))
    while cv2.waitKey() != 27: pass

main()
