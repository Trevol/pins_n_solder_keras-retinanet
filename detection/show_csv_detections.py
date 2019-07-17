import cv2
from utils.VideoController import VideoController
from detection.DetectionsCSV import DetectionsCSV
import detection.visualize


def framePos(videoCapture):
    return int(videoCapture.get(cv2.CAP_PROP_POS_FRAMES))


def main():
    files = [
        ('/HDD_DATA/Computer_Vision_Task/Video_6.mp4',
         './data/detections_video6.csv'),

        ('/HDD_DATA/Computer_Vision_Task/Video_2.mp4',
         './data/detections_video2.csv')
    ]

    for sourceVideoFile, detectionsCsvFile in files:
        videoSource = cv2.VideoCapture(sourceVideoFile)
        ctrl = VideoController(1, state='pause')

        framesDetections = DetectionsCSV.readAsDict(detectionsCsvFile)

        while True:
            pos = framePos(videoSource)
            ret, frame = videoSource.read()
            if not ret:
                break
            detections = framesDetections.get(pos, [])

            detection.visualize.drawDetections(frame, detections)
            detection.visualize.putFramePos(frame, pos)
            cv2.imshow('Video', frame)

            key = ctrl.waitKey()
            if key == 27:
                break

        videoSource.release()


if __name__ == '__main__':
    main()
