import numpy as np
import itertools

from utils.Timer import timeit


def meanSize():
    b1 = (1, 11)
    b3 = [3, 33]
    b5 = [5, 55]

    boxes = np.array([b1, b3, b5])

    meanSize = np.mean(boxes, axis=0)
    print(meanSize)


# meanSize()


dt = np.dtype('2f4,2f4')


def minDist(pts):
    # point-point vectors
    p2p = np.fromiter(itertools.combinations(pts, 2), dt).view(np.float32).reshape(-1, 4)
    vectors = np.subtract(p2p[:, :2], p2p[:, 2:])

    vX = vectors[:, 0]
    vY = vectors[:, 1]
    d = np.sqrt(vX * vX + vY * vY)
    return d.min(), d


def measure():
    pts = np.random.randint(0, 1920, [384, 2])

    for i in range(4):
        with timeit():
            minDist(pts)

    print('----------------------------')
    for i in range(4):
        with timeit():
            for _ in range(100):
                minDist(pts)


measure()
