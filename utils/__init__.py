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
    if factor == 1:
        return img
    w = img.shape[1]
    h = img.shape[0]
    newSize = int(w * factor), int(h * factor)
    return cv2.resize(img, newSize, interpolation=interpolation)


def boxCenter(box, roundPt=False):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    cX = (x1 + x2) / 2
    cY = (y1 + y2) / 2
    if roundPt:
        return roundToInt(cX), roundToInt(cY)
    return cX, cY


def roundToInt(value):
    return int(round(value))


def roundPoint(pt):
    return roundToInt(pt[0]), roundToInt(pt[1])


def cityblockDistance(pt1, pt2):
    x, y = np.abs(np.subtract(pt1, pt2))
    return x + y


def videoWriter(videoCapture: cv2.VideoCapture, videoPath):
    cc = cv2.VideoWriter_fourcc(*'MP4V')  # 'XVID' ('M', 'J', 'P', 'G')
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    w = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (w, h)
    return cv2.VideoWriter(videoPath, cc, fps, size)


def firstOrDefault(items, default=None):
    if any(items):
        return items[0]
    return default


def lastOrDefault(items, default=None):
    if any(items):
        return items[-1]
    return default


def colorChannelsTo24bit(bgr):
    b = int(bgr[0])
    g = int(bgr[1])
    r = int(bgr[2])
    return b + (g << 8) + (r << 16)