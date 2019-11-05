import keras

from detection.RetinanetPinDetector import RetinanetPinDetector
from detection.csv_cache.DetectionsCSV import DetectionsCSV
from models.ModelsContext import ModelsContext
from utils.VideoPlayback import VideoPlayback
from utils.tfSession import get_session

from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image

import utils.visualize

import cv2
import numpy as np


def main():
    files = [
        ('/HDD_DATA/Computer_Vision_Task/Video_6.mp4',
         '/HDD_DATA/Computer_Vision_Task/Video_6_pins_keras_retinanet_detections.avi',
         './data/detections_video6_NEAREST_RESIZING.csv',
         './data/detections_video6_NEAREST_RESIZING.pcl'
         ),

        # ('/HDD_DATA/Computer_Vision_Task/Video_2.mp4',
        #  '/HDD_DATA/Computer_Vision_Task/Video_2_pins_keras_retinanet_detections.avi',
        #  './data/detections_video2.csv')
    ]

    # raise Exception('Attention! Csv files and video files will be overwritten')
    ctx = ModelsContext()
    detector = ctx.getDetector()
    for sourceVideoFile, targetVideoFile, detectionsCsvFile, detectionsPclFile in files:
        video = VideoPlayback(sourceVideoFile)
        # videoTarget = videoWriter(videoSource, targetVideoFile)

        csvWriter = DetectionsCSV(detectionsCsvFile)

        for pos, frame, msec in video.frames():
            _, detections = detector.detect(frame, pos, .5)
            csvWriter.write(pos, detections)
            utils.visualize.drawDetections(frame, detections)
            utils.visualize.putFramePos((10, 40), frame, pos, msec)
            cv2.imshow('Video', frame)
            # videoTarget.write(frame)
            if cv2.waitKey(1) == 27:
                break
        DetectionsCSV.csvToPickle(detectionsCsvFile, detectionsPclFile)

        video.release()
        csvWriter.close()
        # videoTarget.release()


if __name__ == '__main__':
    np.seterr(all='raise')
    main()
