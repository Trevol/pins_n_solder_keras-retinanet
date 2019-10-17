import cv2
import numpy as np

from segmentation.MyVGGUnet import VGGUnet
from segmentation.SceneSegmentation import SceneSegmentation
from utils import remainderlessDividable


class UnetSceneSegmentation(SceneSegmentation):
    class InputBatch:
        imagenetMean = [103.939, 116.779, 123.68]

        def __init__(self, h, w):
            self.w = w
            self.h = h
            self._uintBuf = np.empty((h, w, 3), np.uint8)
            self._floatBuff = np.empty((h, w, 3), np.float32)
            self._imagenetMeans = np.full_like(self._uintBuf, self.imagenetMean, np.float32)

        def prepare(self, image):
            resized = cv2.resize(image, (self.w, self.h), dst=self._uintBuf, interpolation=cv2.INTER_NEAREST)
            normalized = np.subtract(resized, self._imagenetMeans, dtype=np.float32, out=self._floatBuff)
            channelFirst = np.rollaxis(normalized, 2, 0)  # channel_first
            batch = np.expand_dims(channelFirst, 0)  # to batch [1, 3, h, w]
            return batch

        def warmupBatch(self):
            return np.empty([1, 3, self.h, self.w])

    def __init__(self, weightsPath, warmup):
        input_height = remainderlessDividable(1080 // 2, 32, 1)
        input_width = remainderlessDividable(1920 // 2, 32, 1)
        self._inputBatch = self.InputBatch(input_height, input_width)
        self.n_classes = 6
        self.model = VGGUnet(self.n_classes, input_height=input_height, input_width=input_width)
        self.output_height = self.model.outputHeight
        self.output_width = self.model.outputWidth
        self.model.load_weights(weightsPath)

        if warmup:
            self._warmupModel()

    def _warmupModel(self):
        self.model.predict_on_batch(self._inputBatch.warmupBatch())  # warm up model

    def getSegmentationMap(self, frame, framePos):
        batch = self._inputBatch.prepare(frame)
        predictions = self.model.predict(batch)[0]
        pixelClassProbabilities: np.ndarray = predictions.reshape(
            (self.output_height, self.output_width, self.n_classes))
        labelsImage = pixelClassProbabilities.argmax(axis=2)
        return labelsImage
