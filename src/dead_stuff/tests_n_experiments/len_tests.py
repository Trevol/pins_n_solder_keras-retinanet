from more_itertools import ilen
from utils.Timer import Timer


def myLen(iter):
    c = 0
    for _ in iter:
        c += 1
    return c


def myLenWithEnumerate(iter):
    for i, _ in enumerate(iter):
        pass
    return i


def getIter():
    return (p for p in range(1000))


def main():
    iters = 1000

    for _ in range(2):
        with Timer('len list iter').timeit() as ttt:
            for _ in range(iters):
                cnt = len(list(getIter()))

        with Timer('ilen iter').timeit() as ttt:
            for _ in range(iters):
                cnt = ilen(getIter())

        with Timer('myLen iter').timeit() as ttt:
            for _ in range(iters):
                cnt = myLen(getIter())
        with Timer('myLenWithEnumerate iter').timeit() as ttt:
            for _ in range(iters):
                cnt = myLenWithEnumerate(getIter())
        print('-----------------------')


if __name__ == '__main__':
    main()
