import cv2
import numpy as np

from utils import colorChannelsTo24bit


class ColorExtraction:
    @staticmethod
    def rectSelectionMeanColor24(img, rectSelection):
        meanColor = ColorExtraction.rectSelectionAreaMeanColor(img, rectSelection)
        return colorChannelsTo24bit(meanColor)

    @staticmethod
    def rectSelectionCornersMeanColor24(img, rectSelection):
        b24 = colorChannelsTo24bit
        x1, y1 = rectSelection.pt1
        x2, y2 = rectSelection.pt2
        x3, y3 = x1, y2
        x4, y4 = x2, y1
        sumOfColors24 = b24(img[y1, x1]) + b24(img[y2, x2]) + b24(img[y3, x3]) + b24(img[y4, x4])
        return sumOfColors24 // 4

    @staticmethod
    def rectSelectionAreaMeanColor(img, rectSelection):
        x1, y1 = rectSelection.pt1
        x2, y2 = rectSelection.pt2
        area = img[y1:y2 + 1, x1:x2 + 1]
        return np.mean(area, axis=(0, 1))

    @staticmethod
    def multiPointSelectionMeanColor(img, multiPointSelection):
        colors = [img[p[1], p[0]] for p in multiPointSelection.points]
        return np.mean(colors, axis=0)

    @staticmethod
    def multiPointSelectionHsvColors(img, multiPointSelection):
        bgrColors = [img[p[1], p[0]] for p in multiPointSelection.points]

        return [ColorExtraction.bgrTohsv(bgr) for bgr in bgrColors]

    @staticmethod
    def bgrTohsv(bgr):
        return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
