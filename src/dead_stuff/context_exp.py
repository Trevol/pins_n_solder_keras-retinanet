from contextlib import contextmanager


@contextmanager
def context_1():
    print('context_1 enter')
    yield
    print('context_1 exit')


@contextmanager
def context_2():
    print('context_2 enter')
    with context_1():
        yield
    print('context_2 exit')

class Context_3():
    @contextmanager
    def __init__(self):
        print('context_3 enter')
        with context_1():
            yield

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('context_3 exit')

with context_2():
    print('main!!')

print('----------------------')
with Context_3():
    pass
