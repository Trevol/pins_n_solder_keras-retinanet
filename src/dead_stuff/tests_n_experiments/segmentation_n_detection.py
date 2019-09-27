import cv2
import keras
import numpy as np
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image

from segmentation.MyVGGUnet import VGGUnet
from segmentation.classesMeta import BGR
from utils import colorizeLabel, visualize
from utils.tfSession import get_session
from utils import resize, remainderlessDividable


class Segmenter():
    def __init__(self):
        self.input_height = remainderlessDividable(1080 // 2, 32, 1)
        self.input_width = remainderlessDividable(1920 // 2, 32, 1)
        self.n_classes = 6
        self.model = VGGUnet(self.n_classes, input_height=self.input_height, input_width=self.input_width)
        self.output_height = self.model.outputHeight
        self.output_width = self.model.outputWidth

        weights = '../modelWeights/unet_pins_25_0.000016_1.000000.hdf5'
        self.model.load_weights(weights)

    def prepareBatch(self, image):
        image = cv2.resize(image, (self.input_width, self.input_height))
        image = image.astype(np.float32)
        image[:, :, 0] -= 103.939
        image[:, :, 1] -= 116.779
        image[:, :, 2] -= 123.68
        image = np.rollaxis(image, 2, 0)  # channel_first
        return np.expand_dims(image, 0)  # to [1, 3, h, w]

    def segment(self, image):
        batch = self.prepareBatch(image)
        predictions = self.model.predict(batch)[0]
        pixelClassProbabilities: np.ndarray = predictions.reshape((self.output_height, self.output_width, self.n_classes))
        labelsImage = pixelClassProbabilities.argmax(axis=2)
        return labelsImage


class Detector:
    def __init__(self):
        model_path = '../modelWeights/inference_2_28.h5'
        self.model = models.load_model(model_path, backbone_name='resnet50')

    @staticmethod
    def predict_on_image(model, image, scoreThresh):
        image = preprocess_image(image)  # preprocess image for network
        image, scale = resize_image(image)

        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

        boxes = np.divide(boxes, scale, out=boxes)  # boxes /= scale  # correct for image scale

        detections = [(box, label, score) for box, label, score in zip(boxes[0], labels[0], scores[0])
                      if score >= scoreThresh]

        return detections

    def detect(self, image):
        return self.predict_on_image(self.model, image, .5)


def segment_n_predict():
    framePath = '/home/trevol/HDD_DATA/Computer_Vision_Task/frames_6/f_1991_132733.33_132.73.jpg'
    frame = cv2.imread(framePath)

    keras.backend.tensorflow_backend.set_session(get_session())
    segmenter = Segmenter()
    detector = Detector()

    for i in range(100):
        segmentationImage = segmenter.segment(frame)
        detections = detector.detect(frame)

        visDetections = visualize.drawDetections(frame.copy(), detections)
        cv2.imshow('detections', resize(visDetections, .5))

        cv2.imshow('segmentation', colorizeLabel(segmentationImage, BGR))
        cv2.setWindowTitle('segmentation', 'segmentation ' + str(i))

        if cv2.waitKey() == 27:
            break


def main():
    segment_n_predict()


main()
