import cv2
import numpy as np


def imageWithRoi():
    # file, roi = 'messi5.jpg', (20, 273, 121, 295)
    file, roi = '4136.png', (699, 659, 752, 726)
    return cv2.imread(file), roi


def extractRoi(image, roi):
    x1, y1, x2, y2 = roi
    return image[y1:y2, x1:x2]


def drawRoi(img, roi):
    x1, y1, x2, y2 = roi
    return cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 200), 1)


def imshow(*images, **namedImages):
    for wndName, img in enumerate(images):
        cv2.imshow(str(wndName), img)
    for wndName, img in namedImages.items():
        cv2.imshow(wndName, img)


def main():
    bgrImage, roi = imageWithRoi()
    hsvImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2HSV)
    hsvRoiImage = extractRoi(hsvImage, roi)
    roiHsvHist = cv2.calcHist([hsvRoiImage], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(roiHsvHist, roiHsvHist, 0, 255, cv2.NORM_MINMAX)

    backProjection = cv2.calcBackProject([hsvImage], [0, 1], roiHsvHist, [0, 180, 0, 256], 1)
    assert backProjection.dtype == np.uint8

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    backProjection2 = cv2.filter2D(backProjection, -1, disc)

    _, thresholded = cv2.threshold(backProjection2, 40, 255, cv2.THRESH_BINARY)

    result = cv2.bitwise_and(bgrImage, cv2.merge([thresholded, thresholded, thresholded]))
    imshow(drawRoi(bgrImage.copy(), roi), backProjection,
           backProjection2=backProjection2,
           thresholded=thresholded,
           result=result)
    while cv2.waitKey() != 27: pass


def main__():
    bgrImage, roi = imageWithRoi()
    imshow(drawRoi(bgrImage.copy(), roi))
    while cv2.waitKey() != 27: pass


main()
