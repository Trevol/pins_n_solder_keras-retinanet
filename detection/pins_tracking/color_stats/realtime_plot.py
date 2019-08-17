import numpy as np
import matplotlib.pyplot as plt
import cv2

plt.ylim(0, 1)
scatterX = -1


def mouseCallback(evt, x, y, flags, param):
    if evt != cv2.EVENT_LBUTTONDOWN:
        return
    global scatterX
    scatterX += 1
    plt.scatter(scatterX, np.random.random())
    # plt.tight_layout()
    # plt.draw()
    plt.pause(.000001)


cv2.namedWindow('123')
cv2.setMouseCallback('123', mouseCallback)
cv2.waitKey()
