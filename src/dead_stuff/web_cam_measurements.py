from time import clock, perf_counter

import cv2

from utils.Timer import timeit
from utils.VideoPlayback import VideoPlayback


def main():
    video = VideoPlayback(0)
    prevMsec = 0
    t0 = perf_counter()
    for pos, frame, msec in video.frames():
        cv2.imshow('Frame', frame)
        t1 = perf_counter()
        print(pos, msec, msec - prevMsec, (t1 - t0)*1000)
        prevMsec = msec
        t0 = t1
        if cv2.waitKey(1) == 27:
            break


main()
