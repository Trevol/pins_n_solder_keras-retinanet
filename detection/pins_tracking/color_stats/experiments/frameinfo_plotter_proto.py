from detection.pins_tracking.color_stats.FrameInfoPlotter import FrameInfoPlotter
from utils.random_images import random_images
import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def configureLines():
    fig = plt.figure()
    ax = fig.subplots()
    ax.set_ylim(0, 255)
    ax.set_xlim(0, 300)  # initial limit

    bLine = ax.add_line(Line2D([], [], markersize='1', marker='o', linestyle='', color='b'))
    gLine = ax.add_line(Line2D([], [], markersize='1', marker='o', linestyle='', color='g'))
    rLine = ax.add_line(Line2D([], [], markersize='1', marker='o', linestyle='', color='r'))
    return bLine, gLine, rLine


def main():
    hw = 400, 500
    plotter = FrameInfoPlotter(configureLines())

    for i, frame in enumerate(random_images()(hw)):
        cv2.imshow('frame', frame)
        plotter.plot(i, frame[0, 0])
        if cv2.waitKey(50) == 27:
            break


main()
