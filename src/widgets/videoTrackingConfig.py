from detection.RetinanetPinDetector import RetinanetPinDetector
from models.weights.config import retinanet_pins_weights, unet_pins_weights
from segmentation.UnetSceneSegmentation import UnetSceneSegmentation
from techprocess_tracking.TechProcessTracker import TechProcessTracker

videoSource = '/HDD_DATA/Computer_Vision_Task/Video_6.mp4'
# videoSource = '/HDD_DATA/Computer_Vision_Task/Video_2.mp4'
videoSourceDelayMs = 0


# videoSource = 0
# videoSourceDelayMs=-1 #no delay for camera feed
