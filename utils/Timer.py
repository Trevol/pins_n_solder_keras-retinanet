import time
from contextlib import contextmanager


class Timer:
    def __init__(self, desc='', autoreport=True):
        self.desc = desc
        self.start = self.end = self.duration = None
        self.autoreport = autoreport

    @contextmanager
    def timeit(self):
        self.start = time.time()
        yield self
        self.end = time.time()
        self.duration = self.end - self.start
        if self.autoreport:
            print(self.report())

    def report(self):
        return f'{self.desc}: {self.duration:.7f}'


def timeit(desc='', autoreport=True):
    return Timer(desc, autoreport).timeit()

if __name__ == '__main__':
    with timeit('timeit(sleep(1))'):
        time.sleep(1)
    with Timer('Timer().timeit(sleep(1))').timeit():
        time.sleep(1)
