from skimage.color import rgb2hsv
import numpy as np
import cv2

rgb = np.float32([[[55.6, 77.3, 23.78]]])
print(rgb2hsv(rgb))
print(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))