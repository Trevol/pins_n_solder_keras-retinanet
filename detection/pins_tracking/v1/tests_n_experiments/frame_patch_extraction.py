import numpy as np


def main():
    frame = np.ones([20, 20], np.uint8)
    patch = np.float32(frame[5:15, 5:15])
    print(patch)
    patch[3:7, 3:7] = 2
    print(patch)
    print(frame)


main()
