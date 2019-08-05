import itertools
import numpy as np

from utils.Timer import timeit


def measure():
    pts = np.random.randint(0, 1920, [584, 2])
    with timeit():
        combinations = np.array([[x1, y1, x2, y2] for (x1, y1), (x2, y2) in itertools.combinations(pts, 2)])
        # comb = np.array(combinations)
        # print(comb.shape)
    with timeit():
        combinations = np.array([[x1, y1, x2, y2] for (x1, y1), (x2, y2) in itertools.combinations(pts, 2)])

    with timeit():
        combinations = np.array(
            [[comb[0][0], comb[0][1], comb[1][0], comb[1][1]] for comb in itertools.combinations(pts, 2)])

    dt = np.dtype('i,i,i,i')
    with timeit():
        iter = ((comb[0][0], comb[0][1], comb[1][0], comb[1][1]) for comb in itertools.combinations(pts, 2))
        combinations = np.fromiter(iter, dtype=dt, count=-1)

    dt = np.dtype('f,f,f,f')
    with timeit():
        iter = ((comb[0][0], comb[0][1], comb[1][0], comb[1][1]) for comb in itertools.combinations(pts, 2))
        combinations = np.fromiter(iter, dtype=dt, count=-1)

    dt = np.dtype('(2,)f4,(2,)f4')
    with timeit():
        iter = itertools.combinations(pts, 2)
        combinations = np.fromiter(iter, dtype=dt, count=-1)

    with timeit():
        iter = itertools.combinations(pts, 2)
        combinations = np.fromiter(iter, dtype=dt, count=-1)
        combinations = combinations.view(np.float32).reshape(-1, 4)

    print(combinations[0])


def measure():
    pts1 = [
        [1, 1],
        [2, 2]
    ]
    buffer = np.zeros([2, 3], dtype='f4')
    np.power(pts1, 2, out=buffer[:, :2])
    print(buffer)


measure()
