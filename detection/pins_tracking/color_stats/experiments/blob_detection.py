import cv2
import numpy as np


def testImage():
    img = np.full([400, 500], 255, np.uint8)
    # cv2.circle(img, (250, 200), 50, 0, -1)
    cv2.ellipse(img, (250, 200), (50, 70), 30, 0, 360, 0, -1)
    # cv2.rectangle(img, (350, 50), (150, 100), 0, -1)
    return img


def main():
    params = cv2.SimpleBlobDetector_Params()
    params.blobColor=0
    params.filterByArea = False
    params.filterByCircularity = False
    params.filterByColor = False
    params.filterByConvexity = False
    params.filterByInertia = False
    blobDetector = cv2.SimpleBlobDetector_create(params)

    img = testImage()
    keypoints = blobDetector.detect(img)
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, None, (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)


main()
