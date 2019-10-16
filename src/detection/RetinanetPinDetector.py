import cv2
import numpy as np
from keras_retinanet import models
from keras_retinanet.utils.image import compute_resize_scale

from detection.Box import Box
from detection.PinDetector import PinDetector


class RetinanetPinDetector(PinDetector):
    def __init__(self, modelWeightsPath, warmup=True):
        self.model = models.load_model(modelWeightsPath, backbone_name='resnet50')
        self._buffer = None
        self._resizeBuffer = None
        self._imageScale = None
        if warmup:
            self._warmupModel()

    def _warmupModel(self):
        warmupImg = np.zeros([1, 1, 1, 3])
        self.model.predict_on_batch(warmupImg)

    def _preprocess_image(self, img, min_side=800, max_side=1333):
        if self._imageScale is None:
            self._imageScale = compute_resize_scale(img.shape, min_side, max_side)
            self._resizeBuffer = cv2.resize(img, None, dst=None, fx=self._imageScale, fy=self._imageScale)
            self._buffer = np.empty_like(self._resizeBuffer, np.float32)
        else:
            cv2.resize(img, None, dst=self._resizeBuffer, fx=self._imageScale, fy=self._imageScale)

        np.copyto(self._buffer, self._resizeBuffer)
        self._buffer[..., 0] -= 103.939
        self._buffer[..., 1] -= 116.779
        self._buffer[..., 2] -= 123.68
        return self._buffer, self._imageScale

    def predict_on_image(self, model, image, scoreThresh):
        # image, scale = self._preprocess_image(image, min_side=432, max_side=768)
        image, scale = self._preprocess_image(image)

        batch = np.expand_dims(image, axis=0)
        boxes, scores, labels = model.predict_on_batch(batch)

        boxes = np.divide(boxes, scale, out=boxes)  # boxes /= scale  # correct for image scale

        detections = [box_label_score for box_label_score in zip(boxes[0], labels[0], scores[0])
                      if box_label_score[2] >= scoreThresh]

        boxes = (Box(d[0]) for d in detections)
        return boxes, detections

    def detect(self, frame, framePos, scoreThresh):
        return self.predict_on_image(self.model, frame, scoreThresh)
