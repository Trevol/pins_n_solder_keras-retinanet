import cv2
import numpy as np
from collections import deque

from utils.Timer import timeit


def makeFrame(shape=(1080, 1920, 3), value=127):
    return np.full(shape, value, np.uint8)


def manualMeanFrame(framesBuffer):
    pass




def cvMeanFrame(framesBuffer):
    framesBuffer = list(framesBuffer)

    accum = np.float32(framesBuffer[0])
    for frame in framesBuffer[1:]:
        cv2.accumulate(frame, accum)
    mean = np.divide(accum, len(framesBuffer), out=accum)
    return mean

def cvMeanFrame2(framesBuffer):
    framesBuffer = list(framesBuffer)


    accum = np.zeros_like(framesBuffer[0], dtype=np.float32)

    for frame in framesBuffer:
        cv2.accumulate(frame, accum)
    mean = np.divide(accum, len(framesBuffer))
    return mean


def npMeanFrame(framesBuffer):
    np.mean(framesBuffer, axis=0, dtype=np.float32)


def main():
    maxlen = 20
    framesBuffer = deque((makeFrame() for _ in range(maxlen)), maxlen)

    # with timeit('empty func'):
    #     for _ in range(100):
    #         manualMeanFrame(framesBuffer)
    # with timeit('numpy.mean'):
    #     for _ in range(1):
    #         m = npMeanFrame(framesBuffer)
    # with timeit('cvMeanFrame'):
    #     for _ in range(1):
    #         m = cvMeanFrame(framesBuffer)
    # with timeit('cvMeanFrame2'):
    #     for _ in range(1):
    #         m = cvMeanFrame2(framesBuffer)


    with timeit():
        for _ in range(100):
            a = np.zeros([1080, 1920, 3], np.float32)

    with timeit():
        for _ in range(100):
            a.fill(0)

    with timeit():
        for _ in range(100):
            a[:] = 0

    with timeit():
        for _ in range(100):
            a[:, :, :] = 0
    with timeit():
        for _ in range(100):
            a[..., :] = 0


main()
