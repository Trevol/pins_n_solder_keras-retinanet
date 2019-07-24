import numpy as np


class MyClass:
    def __init__(self, value):
        self.value = int(value)

    def __repr__(self):
        return f'MayClass {self.value}'


class MySetFriendlyClass(MyClass):
    def __init__(self, value):
        super(MySetFriendlyClass, self).__init__(value)

    def __hash__(self):
        return self.value


def main():
    randomValues = np.random.randint(0, 20, 20)
    s1 = set(MyClass(v) for v in randomValues)
    print(s1)

    s2 = set(MySetFriendlyClass(v) for v in randomValues)
    print(s2)


main()
