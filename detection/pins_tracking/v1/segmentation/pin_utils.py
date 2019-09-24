import cv2
import numpy as np


def remainderlessDividable(val, divider, ff):
    assert divider > 0
    assert ff == 0 or ff == 1
    return (val + divider * ff - val % divider)


def colorizeLabel(label, colors):
    colorized = np.zeros(label.shape[:2] + (3,), np.uint8)
    for classId, color in enumerate(colors):
        classMask = label == classId
        colorized[..., 0] += np.multiply(classMask, color[0], dtype=np.uint8, casting='unsafe')
        colorized[..., 1] += np.multiply(classMask, color[1], dtype=np.uint8, casting='unsafe')
        colorized[..., 2] += np.multiply(classMask, color[2], dtype=np.uint8, casting='unsafe')
    return colorized


def putLegend(img, names, colors):
    imgHeight = img.shape[0]
    x = 40
    yBase = imgHeight - 40
    yStep = 40
    for i, (name, color) in enumerate(zip(names, colors)):
        y = yBase - i * yStep
        cv2.putText(img, name, (x, y), cv2.FONT_HERSHEY_COMPLEX, .5, color)