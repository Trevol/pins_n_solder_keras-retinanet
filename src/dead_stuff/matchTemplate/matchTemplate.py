import numpy as np
import cv2


class Rect:
    def __init__(self, rect):
        self.rect = rect
        self.pt1, self.pt2 = rect

    def draw(self, img, color=255):
        return cv2.rectangle(img, self.pt1, self.pt2, color, 1)

    def imageCut(self, img):
        (x1, y1), (x2, y2) = self.rect
        return img[y1:y2, x1:x2]


def main():
    im4128 = cv2.imread('4128.png')
    im4136 = cv2.imread('4136.png')

    # rawRect = [(420, 664), (472, 713)]
    rawRect = [(938, 116), (979, 157)]
    templateRect = Rect(rawRect)
    templateFrom4128 = templateRect.imageCut(im4128)

    # cv2.imshow('templateFrom4128', templateFrom4128)
    cv2.imshow('4128', templateRect.draw(im4128.copy()))

    h, w = im4136.shape[:2]
    r = cv2.matchTemplate(im4136[0:h, 0:w], templateFrom4128, cv2.TM_CCORR_NORMED)

    # r = cv2.matchTemplate(im4136, templateFrom4128, cv2.TM_CCORR_NORMED)
    max = r.max()
    print(max)

    x, y = cv2.minMaxLoc(r)[3]
    print(x, y)

    visMatch = (r / max * 255).round().astype(np.uint8)
    visMatch = cv2.circle(visMatch, (x, y), 3, 0, -1)
    cv2.imshow('r', visMatch)
    vis4136 = cv2.circle(templateRect.draw(im4136.copy()), (x, y), 3, (0, 0, 255), -1)
    cv2.imshow('4136', vis4136)

    while cv2.waitKey() != 27: pass


def main():
    im4128 = cv2.imread('4128.png')
    im4136 = np.full_like(im4128, 200)

    # rawRect = [(420, 664), (472, 713)]
    rawRect = [(938, 116), (979, 157)]
    templateRect = Rect(rawRect)
    templateFrom4128 = templateRect.imageCut(im4128)

    r = cv2.matchTemplate(im4136, templateFrom4128, cv2.TM_CCORR_NORMED)
    max = r.max()
    print(max)

    x, y = cv2.minMaxLoc(r)[3]
    print(x, y)

    # cv2.imshow('templateFrom4128', templateFrom4128)
    cv2.imshow('4128', templateRect.draw(im4128.copy()))

    visMatch = (r * 255).round().astype(np.uint8)
    visMatch = cv2.circle(visMatch, (x, y), 3, 0, -1)
    cv2.imshow('r', visMatch)
    vis4136 = cv2.circle(templateRect.draw(im4136.copy()), (x, y), 3, (0, 0, 255), -1)
    cv2.imshow('4136', vis4136)

    while cv2.waitKey() != 27: pass


main()
