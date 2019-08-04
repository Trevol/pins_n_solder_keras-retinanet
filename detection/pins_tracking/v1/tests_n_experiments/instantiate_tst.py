class Ttt:
    def __new__(cls, pins):
        if not any(pins):
            return None
        return object.__new__(cls)

    def __init__(self, pins):
        self.pins = pins

t = Ttt([])
print(t)

t = Ttt([1, 2])
print(t, t.pins)

t = Ttt([3, 4, 5])
print(t, t.pins)
