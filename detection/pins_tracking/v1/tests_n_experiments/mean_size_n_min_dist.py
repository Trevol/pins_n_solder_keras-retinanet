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


def minDist_linalg_norm(pts):
    # point-point vectors
    vectors = np.array([(p1[0] - p2[0], p1[1] - p2[1])
               for i1, p1 in enumerate(pts[:-1])
               for p2 in pts[i1 + 1:]], np.float32)

    d = np.linalg.norm(
        vectors,
        axis=1
    )
    return d.min(), d


def minDist_manual_norm(pts):
    # point-point vectors
    vectors = np.array([(p1[0] - p2[0], p1[1] - p2[1])
                        for i1, p1 in enumerate(pts[:-1])
                        for p2 in pts[i1 + 1:]], np.float32)

    d = np.sqrt(vectors[:, 0] * vectors[:, 0] + vectors[:, 1] * vectors[:, 1])
    return d.min(), d


def minDist_manual_norm_2(pts):
    # point-point vectors
    p2p = np.array([(p1, p2)
                    for i1, p1 in enumerate(pts[:-1])
                    for p2 in pts[i1 + 1:]], np.float32)
    vectors = p2p[:, 0] - p2p[:, 1]

    vX = vectors[:, 0]
    vY = vectors[:, 1]
    d = np.sqrt(vX * vX + vY * vY)
    return d.min(), d


def minDist_manual_norm_3(pts):
    # point-point vectors
    p2p = np.array(list(itertools.combinations(pts, 2)), np.float32)
    vectors = np.subtract(p2p[:, 0], p2p[:, 1])

    vX = vectors[:, 0]
    vY = vectors[:, 1]
    d = np.sqrt(vX * vX + vY * vY)
    return d.min(), d


dt = np.dtype('2f4,2f4')


def minDist_manual_norm_4(pts):
    # point-point vectors
    p2p = np.fromiter(itertools.combinations(pts, 2), dt).view(np.float32).reshape(-1, 4)
    vectors = np.subtract(p2p[:, :2], p2p[:, 2:])

    vX = vectors[:, 0]
    vY = vectors[:, 1]
    d = np.sqrt(vX * vX + vY * vY)
    return d.min(), d


def minDist_manual_norm_5(pts):
    # point-point vectors
    buffer = p2p = np.fromiter(itertools.combinations(pts, 2), dt).view(np.float32).reshape(-1, 4)
    vectors = np.subtract(p2p[:, :2], p2p[:, 2:], out=buffer[:, :2])

    vX = vectors[:, 0]
    vY = vectors[:, 1]
    vX_2 = np.multiply(vX, vX, out=buffer[:, 2])
    vY_2 = np.multiply(vY, vY, out=buffer[:, 3])
    norm_2 = np.add(vX_2, vY_2, out=buffer[:, 1])
    norm = np.sqrt(norm_2, out=buffer[:, 0])
    return norm.min(), norm


p1 = (1, 1)
p2 = (2, 2)
p3 = (3, 3)
pts = [p1, p2, p3]
print(minDist_manual_norm_5(pts))


def measure():
    pts = np.random.randint(0, 1920, [384, 2])

    np.testing.assert_array_equal(
        minDist_linalg_norm(pts)[1],
        minDist_manual_norm(pts)[1]
    )

    np.testing.assert_array_equal(
        minDist_linalg_norm(pts)[1],
        minDist_manual_norm_2(pts)[1]
    )
    np.testing.assert_array_equal(
        minDist_linalg_norm(pts)[1],
        minDist_manual_norm_3(pts)[1]
    )
    np.testing.assert_array_equal(
        minDist_linalg_norm(pts)[1],
        minDist_manual_norm_4(pts)[1]
    )
    np.testing.assert_array_equal(
        minDist_linalg_norm(pts)[1],
        minDist_manual_norm_5(pts)[1]
    )


    for i in range(4):
        with timeit():
            minDist_linalg_norm(pts)
    print('----------------------------')
    for i in range(4):
        with timeit():
            minDist_manual_norm(pts)
    print('----------------------------')
    for i in range(4):
        with timeit():
            minDist_manual_norm_2(pts)
    print('----------------------------')
    for i in range(4):
        with timeit():
            minDist_manual_norm_3(pts)
    print('----------------------------')
    for i in range(4):
        with timeit():
            minDist_manual_norm_4(pts)
    print('----------------------------')
    for i in range(4):
        with timeit():
            minDist_manual_norm_5(pts)

    print('----------------------------')
    for i in range(4):
        with timeit():
            for _ in range(100):
                minDist_manual_norm_4(pts)
    print('----------------------------')
    for i in range(4):
        with timeit():
            for _ in range(100):
                minDist_manual_norm_5(pts)


measure()
