import cv2
import numpy as np

from segmentation.MyVGGUnet import VGGUnet
from segmentation.SceneSegmentation import SceneSegmentation
from utils import remainderlessDividable


class UnetSceneSegmentation(SceneSegmentation):
    def __init__(self, weightsPath, warmup):
        self.input_height = remainderlessDividable(1080 // 2, 32, 1)
        self.input_width = remainderlessDividable(1920 // 2, 32, 1)
        self.n_classes = 6
        self.model = VGGUnet(self.n_classes, input_height=self.input_height, input_width=self.input_width)
        self.output_height = self.model.outputHeight
        self.output_width = self.model.outputWidth
        self.model.load_weights(weightsPath)
        if warmup:
            self._warmupModel()

    def _warmupModel(self):
        img = np.zeros([1, 3, self.input_height, self.input_width])
        self.model.predict_on_batch(img)  # warm up model

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
