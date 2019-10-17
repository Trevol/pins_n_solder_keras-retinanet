import cv2
import numpy as np
from keras_retinanet import models
from keras_retinanet.utils.image import compute_resize_scale

from detection.Box import Box
from detection.PinDetector import PinDetector


class RetinanetPinDetector(PinDetector):
    class InputBatch:
        imagenetMean = [103.939, 116.779, 123.68]

        def __init__(self, imageShape, min_side=800, max_side=1333):
            self.scale = compute_resize_scale(imageShape, min_side, max_side)
            h = round(imageShape[0] * self.scale)
            w = round(imageShape[1] * self.scale)
            self._resizeBuffer = np.empty((h, w, 3), np.uint8)
            self._buffer = np.empty_like(self._resizeBuffer, np.float32)
            self._imagenetMeans = np.full_like(self._resizeBuffer, self.imagenetMean, np.float32)

        def prepare(self, img):
            resized = cv2.resize(img, None, dst=self._resizeBuffer, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
            normalized = np.subtract(resized, self._imagenetMeans, dtype=np.float32, out=self._buffer)
            batch = np.expand_dims(normalized, axis=0)
            return batch

        @staticmethod
        def warmupBatch():
            return np.zeros([1, 1, 1, 3])

        def rescaleBoxes(self, boxes, out=None):
            return np.divide(boxes, self.scale, out=out)

    def __init__(self, modelWeightsPath, warmup=True):
        self._inputBatch = None
        self.model = models.load_model(modelWeightsPath, backbone_name='resnet50')
        if warmup:
            self._warmupModel()

    def _warmupModel(self):
        self.model.predict_on_batch(self.InputBatch.warmupBatch())

    def predict_on_image(self, model, image, scoreThresh):
        if self._inputBatch is None:
            self._inputBatch = self.InputBatch(image.shape, min_side=800, max_side=1333)  # min_side=432, max_side=768
        batch = self._inputBatch.prepare(image)
        boxes, scores, labels = model.predict_on_batch(batch)

        boxes = self._inputBatch.rescaleBoxes(boxes, out=boxes)
        detections = [box_label_score for box_label_score in zip(boxes[0], labels[0], scores[0])
                      if box_label_score[2] >= scoreThresh]
        boxes = (Box(d[0]) for d in detections)
        return boxes, detections

    def detect(self, frame, framePos, scoreThresh):
        return self.predict_on_image(self.model, frame, scoreThresh)
