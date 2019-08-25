import numpy as np
import cv2


def main():
    rgb = np.float32([69, 200, 43]).reshape(1, 1, 3)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    print(hsv)


if __name__ == '__main__':
    main()
