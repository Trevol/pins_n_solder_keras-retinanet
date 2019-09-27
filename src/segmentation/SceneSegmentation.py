import glob
import os
import cv2
import numpy as np

from segmentation.MyVGGUnet import VGGUnet
from utils import remainderlessDividable


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


class UnetSceneSegmentation(SceneSegmentation):
    def __init__(self, weightsPath):
        self.input_height = remainderlessDividable(1080 // 2, 32, 1)
        self.input_width = remainderlessDividable(1920 // 2, 32, 1)
        self.n_classes = 6
        self.model = VGGUnet(self.n_classes, input_height=self.input_height, input_width=self.input_width)
        self.output_height = self.model.outputHeight
        self.output_width = self.model.outputWidth
        self.model.load_weights(weightsPath)

    def prepareBatch(self, image):
        image = cv2.resize(image, (self.input_width, self.input_height))
        image = image.astype(np.float32)
        image[:, :, 0] -= 103.939
        image[:, :, 1] -= 116.779
        image[:, :, 2] -= 123.68
        image = np.rollaxis(image, 2, 0)  # channel_first
        return np.expand_dims(image, 0)  # to batch [1, 3, h, w]

    def getSegmentationMap(self, frame, framePos):
        batch = self.prepareBatch(frame)
        predictions = self.model.predict(batch)[0]
        pixelClassProbabilities: np.ndarray = predictions.reshape(
            (self.output_height, self.output_width, self.n_classes))
        labelsImage = pixelClassProbabilities.argmax(axis=2)
        return labelsImage


if __name__ == '__main__':
    cacheDirectory = '/home/trevol/HDD_DATA/Computer_Vision_Task/frames_6/not_augmented_base_vgg16_more_images_25'
    cache = CachedSceneSegmentation(cacheDirectory)
    assert cache.getSegmentationMap(None, 100) is not None
