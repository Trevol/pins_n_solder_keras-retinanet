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
        return f'{self.desc}: {self.duration}'

    def __repr__(self):
        return self.report()