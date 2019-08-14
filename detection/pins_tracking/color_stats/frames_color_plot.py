import matplotlib.pyplot as plt
import numpy as np


def to32bit(bgr):
    b = bgr[..., 0]
    g = bgr[..., 1]
    r = bgr[..., 2]
    result = b + np.left_shift(g, 8, dtype=np.int32) + np.left_shift(g, 16, dtype=np.int32)
    return result


def plotBGR(file):
    frameBGR = np.load(file)
    frames = frameBGR[:, 0]
    b = frameBGR[:, 1]
    g = frameBGR[:, 2]
    r = frameBGR[:, 3]

    plt.scatter(frames, b)
    plt.scatter(frames, g)
    plt.scatter(frames, r)
    plt.tight_layout()
    plt.show()

def plot32bit(file):
    frameBGR = np.load(file)
    frames = frameBGR[:, 0]
    bgr = frameBGR[..., 1:4]
    color32bit = to32bit(bgr)

    plt.scatter(frames, color32bit, s=1)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # file = 'frame_colors_1276_733.npy'
    file = 'frame_colors_1333_186.npy'
    plot32bit(file)
