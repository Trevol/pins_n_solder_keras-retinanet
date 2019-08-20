import numpy as np

from utils import roundToInt


class random_images:
    @staticmethod
    def rndIntensities():
        def next_(current, direction, step):
            current = current + step * direction
            if current <= from_ or current >= to:
                direction = -direction
                if current <= from_:
                    current = from_
                else:
                    current = to
            return current, direction

        from_, to = 0, 255
        current = np.random.randint(from_, to + 1)
        step = (np.random.random() + .5) * 1.5
        direction = 1 if np.random.randint(0, 2) == 0 else -1
        while True:
            yield roundToInt(current)
            current, direction = next_(current, direction, step)

    def __call__(self):
        channel = self.rndIntensities
        colors = zip(channel(), channel(), channel())
        for color in colors:
            img = np.empty([400, 500, 3], np.uint8)
            img[:, :] = color
            yield img