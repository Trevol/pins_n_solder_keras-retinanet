import cv2
import numpy as np


def clip(value, x1, x2):
    if value < x1:
        return x1
    if value > x2:
        return x2
    return value


def leftClip(value, x1):
    if value < x1:
        return x1
    return value


def resize(img, factor, interpolation=cv2.INTER_AREA):
    dsize = tuple(np.multiply(img.shape[1::-1], factor).astype(int))
    return cv2.resize(img, dsize, interpolation=interpolation)
