import cv2
from models.detection import DetectionsCSV
import utils
from utils import resize
from utils import VideoPlayback


def csvToPcl():
    DetectionsCSV.csvToPickle('./data/detections_video6.csv', './data/detections_video6.pcl')
    DetectionsCSV.csvToPickle('./data/detections_video2.csv', './data/detections_video2.pcl')


def files():
    yield ('/HDD_DATA/Computer_Vision_Task/Video_6.mp4',
           DetectionsCSV.loadPickle('./data/detections_video6.pcl'))

    yield ('/HDD_DATA/Computer_Vision_Task/Video_2.mp4',
           DetectionsCSV.loadPickle('./data/detections_video2.pcl'))


def main():
    winname = 'Video'

    def indicatePlaybackState(frameDelay, autoPlay, framePos, framePosMsec, playback):
        autoplayLabel = 'ON' if autoPlay else 'OFF'
        stateTitle = f'{winname} (FrameDelay: {frameDelay}, Autoplay: {autoplayLabel})'
        cv2.setWindowTitle(winname, stateTitle)

    def frameReady(frame, framePos, framePosMsec, playback):
        detections = framesDetections.get(framePos, [])
        utils.visualize.drawDetections(frame, detections)
        utils.visualize.putFramePos(frame, framePos)
        if frame.shape[1] >= 1900:
            frame = resize(frame, .8)
        cv2.imshow(winname, frame)

    for sourceVideoFile, framesDetections in files():
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=False)
        videoPlayback.play(onFrameReady=frameReady, onStateChange=indicatePlaybackState)
        videoPlayback.release()


if __name__ == '__main__':
    main()
