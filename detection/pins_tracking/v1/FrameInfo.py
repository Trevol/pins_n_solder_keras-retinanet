import numpy as np

from detection.pins_tracking.v1.Box import Box


class FrameInfo:
    def __init__(self, bboxes, pos, posMsec, frame):
        self.pos = pos
        self.posMsec = posMsec
        self.bboxes = bboxes
        # TODO: extract frame patches for bboxes
        # self.framePatch = self.__extractPatches(frame, bboxes)

    def boxByPoint(self, pt):
        return Box.boxByPoint(self.bboxes, pt)

    @classmethod
    def __extractPatches(cls, frame, bboxes):
        return [cls.__extractPatch(frame, b) for b in bboxes]

    @staticmethod
    def __extractPatch(frame, bbox):
        patchBox = np.ceil(bbox.box).astype(np.int32) + 2
        x0, y0, x1, y1 = patchBox
        patch = frame[y0:y1, x0:x1].copy()
        return patch