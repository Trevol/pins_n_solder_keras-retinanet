import numpy as np
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image

from detection.Box import Box
from detection.PinDetector import PinDetector


class RetinanetPinDetector(PinDetector):
    def __init__(self, modelWeightsPath, warmup=True):
        self.model = models.load_model(modelWeightsPath, backbone_name='resnet50')
        if warmup:
            self._warmupModel()

    def _warmupModel(self):
        warmupImg = np.zeros([1, 1, 1, 3])
        self.model.predict_on_batch(warmupImg)

    @staticmethod
    def predict_on_image(model, image, scoreThresh):
        image = preprocess_image(image)  # preprocess image for network
        # image, scale = resize_image(image, min_side=432, max_side=768)
        image, scale = resize_image(image)

        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

        boxes = np.divide(boxes, scale, out=boxes)  # boxes /= scale  # correct for image scale

        detections = [(box, label, score) for box, label, score in zip(boxes[0], labels[0], scores[0])
                      if score >= scoreThresh]
        boxes = (Box(d[0]) for d in detections)
        return boxes, detections

    def detect(self, frame, framePos, scoreThresh):
        return self.predict_on_image(self.model, frame, scoreThresh)
