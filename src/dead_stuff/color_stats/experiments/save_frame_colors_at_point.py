import numpy as np

from dead_stuff.color_stats.main_color_stats import files
from utils import VideoPlayback


def save_frame_colors_at_point():
    def flipXY(*points):
        return [p[::-1] for p in points]

    list_yxOfInterest = flipXY((951, 186), (1233, 196), (1333, 186))
    frameColorsAtPoint = [[] for _ in list_yxOfInterest]

    for sourceVideoFile in files():
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=True)

        for pos, frame, _ in videoPlayback.frames():
            for index, yxOfInterest in enumerate(list_yxOfInterest):
                row = [pos, *frame[yxOfInterest]]
                frameColorsAtPoint[index].append(row)

        for yxOfInterest, colors in zip(list_yxOfInterest, frameColorsAtPoint):
            np.save(f'frame_colors_{yxOfInterest[1]}_{yxOfInterest[0]}.npy', colors)
        videoPlayback.release()