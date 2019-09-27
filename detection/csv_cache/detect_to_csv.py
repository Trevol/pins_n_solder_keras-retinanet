import keras
from utils.tfSession import get_session

from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image

import utils.visualize

import cv2
import numpy as np


def main():
    keras.backend.tensorflow_backend.set_session(get_session())
    model_path = '/HDD_DATA/training_checkpoints/keras_retinanet/pins/2/inference_2_28.h5'
    model = models.load_model(model_path, backbone_name='resnet50')

    # load label to names mapping for visualization purposes
    labels_to_names = {0: 'pin', 1: 'solder'}

    files = [
        ('/HDD_DATA/Computer_Vision_Task/Video_6.mp4',
         '/HDD_DATA/Computer_Vision_Task/Video_6_pins_keras_retinanet_detections.avi',
         './data/detections_video6.csv'),

        ('/HDD_DATA/Computer_Vision_Task/Video_2.mp4',
         '/HDD_DATA/Computer_Vision_Task/Video_2_pins_keras_retinanet_detections.avi',
         './data/detections_video2.csv')
    ]

    # raise Exception('Attention! Csv files and video files will be overwritten')

    for sourceVideoFile, targetVideoFile, detectionsCsvFile in files:
        videoSource = cv2.VideoCapture(sourceVideoFile)
        # videoTarget = videoWriter(videoSource, targetVideoFile)

        # csvWriter = DetectionsCSV(detectionsCsvFile)

        framePos = 0
        while True:
            ret, frame = videoSource.read()
            if not ret:
                break
            detections = predict_on_image(model, frame, labels_to_names, scoreThresh=0.5)
            # csvWriter.write(framePos, detections)
            utils.visualize.drawDetections(frame, detections)
            utils.visualize.putFramePos(frame, framePos)
            cv2.imshow('Video', frame)
            # videoTarget.write(frame)
            if cv2.waitKey(1) == 27:
                break
            framePos += 1

        videoSource.release()
        # videoTarget.release()
        # csvWriter.close()


def predict_on_image(model, image, labels_to_names, scoreThresh):
    image = preprocess_image(image)  # preprocess image for network
    image, scale = resize_image(image)

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    boxes = np.divide(boxes, scale, out=boxes)  # boxes /= scale  # correct for image scale

    detections = [(box, label, score) for box, label, score in zip(boxes[0], labels[0], scores[0])
                  if score >= scoreThresh]

    return detections


if __name__ == '__main__':
    main()
