import cv2
from detection.csv_cache.DetectionsCSV import DetectionsCSV
import utils.visualize
from utils.VideoPlayback import VideoPlayback


def csvToPcl():
    DetectionsCSV.csvToPickle('./data/detections_video6.csv', './data/detections_video6.pcl')
    DetectionsCSV.csvToPickle('./data/detections_video2.csv', './data/detections_video2.pcl')


def main():
    winname = 'Video'

    def indicatePlaybackState(frameDelay, autoPlay, framePos, playback):
        autoplayLabel = 'ON' if autoPlay else 'OFF'
        stateTitle = f'{winname} (FrameDelay: {frameDelay}, Autoplay: {autoplayLabel})'
        cv2.setWindowTitle(winname, stateTitle)

    files = [
        ('/HDD_DATA/Computer_Vision_Task/Video_6.mp4',
         DetectionsCSV.loadPickle('./data/detections_video6.pcl')),

        ('/HDD_DATA/Computer_Vision_Task/Video_2.mp4',
         DetectionsCSV.loadPickle('./data/detections_video2.pcl'))
    ]

    def frameReady(frame, framePos, playback):
        detections = framesDetections.get(framePos, [])
        utils.visualize.drawDetections(frame, detections)
        utils.visualize.putFramePos(frame, framePos)
        cv2.imshow(winname, frame)

    for sourceVideoFile, framesDetections in files:
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=False)
        videoPlayback.play(onFrameReady=frameReady, onStateChange=indicatePlaybackState)
        videoPlayback.release()


if __name__ == '__main__':
    main()
