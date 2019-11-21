import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    img = np.ones([4, 4], np.uint8)
    img2 = np.full([4, 4], 3, np.uint8)
    h = cv2.calcHist([img2, img], channels=[0, 1], mask=None, histSize=[4, 4], ranges=[0, 4, 0, 4])
    p = plt.imshow(h)
    plt.colorbar(p)
    plt.show()


main()
