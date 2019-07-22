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


def boxCenter(box, roundToInt=False):
    pt1 = np.float32([box[0], box[1]])
    pt2 = [box[2], box[3]]
    buffer = np.add(pt1, pt2, out=pt1)
    result = np.divide(buffer, 2, out=buffer)
    if roundToInt:
        result = np.round(result, 0, out=result).astype(np.int32)
    return result


def cityblockDistance(pt1, pt2):
    x, y = np.abs(np.subtract(pt1, pt2))
    return x + y
