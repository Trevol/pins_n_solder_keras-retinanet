from detection.PinDetector import PickledDictionaryPinDetector, RetinanetPinDetector
from segmentation.SceneSegmentation import CachedSceneSegmentation, UnetSceneSegmentation
from techprocess_tracking.TechProcessTracker import TechProcessTracker

videoSource = '/HDD_DATA/Computer_Vision_Task/Video_6.mp4'
# videoSource = '/HDD_DATA/Computer_Vision_Task/Video_2.mp4'
videoSourceDelayMs = 0

# videoSource = 0
# videoSourceDelayMs=-1 #no delay for camera feed

def techProcessTrackerFactory():
    # pinDetector = PickledDictionaryPinDetector('detection/csv_cache/data/detections_video6.pcl')
    # sceneSegmentation = CachedSceneSegmentation(
    #     '/home/trevol/HDD_DATA/Computer_Vision_Task/frames_6/not_augmented_base_vgg16_more_images_25')

    pinDetector = RetinanetPinDetector('modelWeights/retinanet_pins_inference.h5')
    sceneSegmentation = UnetSceneSegmentation('modelWeights/unet_pins_25_0.000016_1.000000.hdf5')

    techProcessTracker = TechProcessTracker(pinDetector, sceneSegmentation)
    return techProcessTracker
