import cv2
import numpy as np


def testImage():
    img = np.full([400, 500], 0, np.uint8)
    # cv2.circle(img, (250, 200), 50, 0, -1)
    cv2.ellipse(img, (150, 100), (50, 70), 30, 0, 360, 255, -1)
    # cv2.rectangle(img, (350, 50), (150, 100), 0, -1)
    return img


def makeSimpleDetector():
    params = cv2.SimpleBlobDetector_Params()
    params.blobColor = 127
    params.filterByColor = True
    params.filterByArea = False
    params.filterByCircularity = False
    params.filterByColor = False
    params.filterByConvexity = False
    params.filterByInertia = False
    return cv2.SimpleBlobDetector_create(params)


def main():
    img = testImage()

    detectors = [
        makeSimpleDetector(),
        # cv2.MSER_create(),
        # cv2.AgastFeatureDetector_create(),
        # cv2.FastFeatureDetector_create(),
        # cv2.ORB_create()
    ]
    for detector in detectors:
        detector.getDefaultName()
        keypoints = detector.detect(img)
        im_with_keypoints = cv2.drawKeypoints(img, keypoints, None, (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow(f"{detector.getDefaultName()} Keypoints", im_with_keypoints)

    cv2.waitKey()


main()
