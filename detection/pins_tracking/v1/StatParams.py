import numpy as np


class StatParams:
    def __init__(self, mean, std, median):
        self.mean = mean
        self.std = std
        self.median = median

    def areFromDifferentDistributions(self, other):
        # absdiff = np.abs(colorStat1.mean - colorStat2.mean)
        # thresh = colorStat1.std * 3
        absdiff = np.abs(self.median - other.median)
        soDifferent = np.any(absdiff > 14.5)  # at least one component is outside of std deviation
        return soDifferent
