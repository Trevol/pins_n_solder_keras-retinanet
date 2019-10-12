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


with context_2():
    print('main!!')
