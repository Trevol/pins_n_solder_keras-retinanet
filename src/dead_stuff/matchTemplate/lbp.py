from skimage import feature
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import resize
from utils.Timer import timeit


def main():
    img = cv2.imread('4136.png', cv2.IMREAD_GRAYSCALE)
    numPoints = 25
    radius = 8
    lbp = feature.local_binary_pattern(img, numPoints, radius, method='uniform')
    # bin for each unique value in lbp - range [0, numPoints+1]
    hist, binEdges = np.histogram(np.ravel(lbp), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))


def main__():
    img = np.full([4, 4], 127, np.uint8)
    hist = cv2.calcHist([img], [0], mask=None, histSize=[10], ranges=[0, 256])
    histNumpy, edges = np.histogram(img, 10, range=[0, 256])
    plt.hist(np.ravel(img), bins=256, range=[0, 256])

    plt.show()


main()
