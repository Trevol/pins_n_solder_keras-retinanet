import numpy as np
import cv2


def intersection(box1, box2):
    return 1


def union(box1, box2):
    return cv2.intersectConvexConvex()


def iou(box1, box2):
    return intersection(box1, box2) / union(box1, box2)


def main():
    b1 = np.array([
        [1, 1],
        [1, 3],
        [3, 3],
        [3, 1]
    ])
    b2 = np.array([
        [2, 2],
        [4, 2],
        [4, 4],
        [2, 4]
    ])

    retval, intersection = cv2.intersectConvexConvex(b1, b2)
    print(retval, intersection)

    retval, intersection = cv2.intersectConvexConvex(b1, b1)
    print(retval, intersection)

    # b1 = (
    #     [1., 1.],
    #     [3., 3.],
    #     0
    # )
    # b2 = (
    #     [2., 2.],
    #     [4., 4.],
    #     0
    # )
    #
    # retval, region = cv2.rotatedRectangleIntersection(b1, b2, None)
    # print(retval, region)

main()
