import cv2


def main():
    files = [
        ('/HDD_DATA/Computer_Vision_Task/Video_6.mp4',
         './data/detections_video6.csv'),

        ('/HDD_DATA/Computer_Vision_Task/Video_2.mp4',
         './data/detections_video2.csv')
    ]

    for sourceVideoFile, detectionsCsvFile in files:
        videoSource = cv2.VideoCapture(sourceVideoFile)

        from detection.DetectionsCSV import DetectionsCSV
        import detection.visualize

        framesDetections = DetectionsCSV.readAsDict(detectionsCsvFile)

        framePos = 0
        while True:
            ret, frame = videoSource.read()
            if not ret:
                break
            detections = framesDetections.get(framePos, [])

            detection.visualize.drawDetections(frame, detections)
            detection.visualize.putFramePos(frame, framePos)
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) == 27:
                break
            framePos += 1

        videoSource.release()


if __name__ == '__main__':
    main()
