import glob
import os
import cv2


class SceneSegmentation:
    def getSegmentationMap(self, frame, framePos): pass


class CachedSceneSegmentation(SceneSegmentation):
    def __init__(self, cacheDirectory):
        labelPaths = sorted(glob.glob(os.path.join(cacheDirectory, '*_label.png')))
        self.cachedLabelsPaths = [(self.framePosFromPath(p), p) for p in labelPaths]

    @staticmethod
    def framePosFromPath(filePath):
        # fileName example f_0002_133.33_0.13_label.png
        fileName = os.path.basename(filePath)
        framePosPart = fileName.split('_', maxsplit=2)[1]
        return int(framePosPart) - 1

    def getSegmentationMap(self, frame, framePos):
        item = self.cachedLabelsPaths[framePos]
        assert item[0] == framePos
        path = item[1]
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


if __name__ == '__main__':
    cacheDirectory = '/home/trevol/HDD_DATA/Computer_Vision_Task/frames_6/not_augmented_base_vgg16_more_images_25'
    cache = CachedSceneSegmentation(cacheDirectory)
    assert cache.getSegmentationMap(None, 100) is not None
