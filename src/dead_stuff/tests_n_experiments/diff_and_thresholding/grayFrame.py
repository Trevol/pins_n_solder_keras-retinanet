import cv2

from utils import VideoPlayback


def getConsequentFrames():
    # frames = 8292, 8361 # solder added
    frames = 3667, 3716
    # frames = 2436, 2633
    # frames = (420, 586)
    p = VideoPlayback('/HDD_DATA/Computer_Vision_Task/Video_6.mp4', 1, autoplayInitially=False)
    f1 = p.readFrame(frames[0])
    f2 = p.readFrame(frames[1])
    return f1, f2


def makeDiff(f1, f2):
    # TODO: compute gradients and try extract peaks/ridges(хребты). By ridgeDetector?

    # наши "эллипсы" самые светлые - свет падает на них и хорошо отражается в камеру
    # cv2.subtract отриц. значения (более темные) устанавливает в 0, что и требуется
    colorDiff = cv2.subtract(f2, f1)

    return colorDiff


def showBlobsOnDiff():
    def s(f):
        # return f
        return f[100:, 350:1600]

    f1, f2 = getConsequentFrames()


    # show results
    grayF2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    cv2.imshow('f2', s(grayF2))

    while cv2.waitKey() != 27: pass


def main():
    # TODO: detect blobs on mask and grayDiff(with thresholding)
    showBlobsOnDiff()
    # playVideo()


main()
