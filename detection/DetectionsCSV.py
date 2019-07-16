import csv
from itertools import groupby
import numpy as np


class DetectionsCSV:
    def __init__(self, csvPath):
        self.file = open(csvPath, mode='w')
        self.rowsCount = 0

    def write(self, framePos, detections):
        for detection in detections:
            box, label, score = detection
            x1, y1, x2, y2 = box
            row = f'{framePos},{x1},{y1},{x2},{y2},{label},{score}'
            if self.rowsCount > 0:
                self.file.write('\n')
            self.file.write(row)
            self.rowsCount += 1

    def close(self):
        self.file.flush()
        self.file.close()

    @staticmethod
    def toTypedRow(csvRow):
        framePos, x1, y1, x2, y2, label, score = csvRow
        return int(framePos), np.float32([x1, y1, x2, y2]), int(label), float(score)

    @staticmethod
    def framePosFn(row):
        return row[0]

    @staticmethod
    def skipFramePos(rows):
        return [r[1:] for r in rows]

    @classmethod
    def readAsDict(cls, csvPath):
        with open(csvPath) as file:
            reader = csv.reader(file, delimiter=',')
            reader = map(cls.toTypedRow, reader)
            reader = sorted(reader, key=cls.framePosFn)
            reader = ((framePos, cls.skipFramePos(rows)) for framePos, rows in groupby(reader, key=cls.framePosFn))
            framesDetections = dict(reader)
            return framesDetections
