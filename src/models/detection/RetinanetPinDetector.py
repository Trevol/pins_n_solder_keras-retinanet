import numpy as np
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image

from models.detection import Box
from models.detection.PinDetector import PinDetector


class RetinanetPinDetector(PinDetector):
    def __init__(self, modelWeightsPath):
        self.model = models.load_model(modelWeightsPath, backbone_name='resnet50')

    @staticmethod
    def predict_on_image(model, image, scoreThresh):
        image = preprocess_image(image)  # preprocess image for network
        image, scale = resize_image(image)

        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

        boxes = np.divide(boxes, scale, out=boxes)  # boxes /= scale  # correct for image scale

        detections = [(box, label, score) for box, label, score in zip(boxes[0], labels[0], scores[0])
                      if score >= scoreThresh]
        boxes = (Box(d[0]) for d in detections)
        return boxes, detections

    def detect(self, frame, framePos, scoreThresh):
        return self.predict_on_image(self.model, frame, scoreThresh)