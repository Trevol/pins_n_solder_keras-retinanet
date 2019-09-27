import numpy as np
import cv2

import utils.visualize
from detection.PinDetector import PinDetector, PickledDictionaryPinDetector, RetinanetPinDetector
from segmentation.SceneSegmentation import CachedSceneSegmentation, UnetSceneSegmentation
from utils.VideoPlayback import VideoPlayback

from techprocess_tracking.TechProcessTracker import TechProcessTracker
from utils.VideoPlaybackHandlerBase import VideoPlaybackHandlerBase


class TechProcessVideoHandler(VideoPlaybackHandlerBase):
    def __init__(self, frameSize, pinDetector: PinDetector, sceneSegmentation):
        super(TechProcessVideoHandler, self).__init__(frameSize)
        self.techProcessTracker = TechProcessTracker(pinDetector, sceneSegmentation)

    def frameReady(self, frame, framePos, framePosMsec, playback):
        self.techProcessTracker.track(framePos, framePosMsec, frame)
        super(TechProcessVideoHandler, self).frameReady(frame, framePos, framePosMsec, playback)

    def processDisplayFrame(self, displayFrame0):
        utils.visualize.putFramePos((10, 40), displayFrame0, self._framePos, self._framePosMsec)
        self.drawStats(self.techProcessTracker.getStats(), displayFrame0, (10, 110))
        self.techProcessTracker.drawScene(displayFrame0)
        return displayFrame0

    def drawStats(self, stats, onImage, atPoint):
        if stats is None:
            return
        pinsCount, pinsWithSolderCount = stats
        red = (0, 0, 255)
        text = f'Pins: {pinsCount}'
        cv2.putText(onImage, text, atPoint, cv2.FONT_HERSHEY_COMPLEX, .7, red)
        x, y = atPoint
        text = f'Solder: {pinsWithSolderCount}'
        cv2.putText(onImage, text, (x, y + 30), cv2.FONT_HERSHEY_COMPLEX, .7, red)

    def release(self):
        super(TechProcessVideoHandler, self).release()


def files():
    yield ('/HDD_DATA/Computer_Vision_Task/Video_6.mp4',
           '/HDD_DATA/Computer_Vision_Task/Video_6_result.mp4',
           'detection/csv_cache/data/detections_video6.pcl',
           '/home/trevol/HDD_DATA/Computer_Vision_Task/frames_6/not_augmented_base_vgg16_more_images_25')

    # yield ('/HDD_DATA/Computer_Vision_Task/Video_2.mp4',
    #        '/HDD_DATA/Computer_Vision_Task/Video_2_result.mp4',
    #        '../../csv_cache/data/detections_video2.pcl')


def printMemoryUsage():
    # print(psutil.Process().memory_info())  # in bytes
    pass


def main():
    printMemoryUsage()

    def getFramesRange():
        # framesRange = (4150, None)
        # framesRange = (8100, None)
        framesRange = None
        return framesRange

    np.seterr(all='raise')
    for sourceVideoFile, resultVideo, pclFile, segmentationCacheDir in files():
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=False)

        pinDetector = PickledDictionaryPinDetector(pclFile)
        pinDetector = RetinanetPinDetector('modelWeights/retinanet_pins_inference.h5')

        sceneSegmentation = CachedSceneSegmentation(segmentationCacheDir)
        # sceneSegmentation = UnetSceneSegmentation('modelWeights/unet_pins_25_0.000016_1.000000.hdf5')

        handler = TechProcessVideoHandler(videoPlayback.frameSize(), pinDetector, sceneSegmentation)

        endOfVideo = videoPlayback.playWithHandler(handler, range=getFramesRange())
        printMemoryUsage()

        if endOfVideo:
            cv2.waitKey()

        videoPlayback.release()
        handler.release()

    cv2.waitKey()


if __name__ == '__main__':
    main()
