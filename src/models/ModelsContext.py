from threading import Lock

import gc
import tensorflow as tf
import keras.backend as K
from Cython.Compiler.TypeSlots import GCClearReferencesSlot

from detection.PickledDictionaryPinDetector import PickledDictionaryPinDetector
from detection.RetinanetPinDetector import RetinanetPinDetector
from models.weights.config import retinanet_pins_weights, unet_pins_weights
from segmentation.CachedSceneSegmentation import CachedSceneSegmentation
from segmentation.UnetSceneSegmentation import UnetSceneSegmentation


class ModelsContext:
    __detector = None
    __segmentation = None
    session = None
    graph = None
    _lock = Lock()

    def __init__(self):
        self._initializer()

    @classmethod
    def getDetector(cls):
        cls._initializer()
        return cls.__detector

    @classmethod
    def getSegmentation(cls):
        cls._initializer()
        return cls.__segmentation

    @classmethod
    def _create(cls):
        with cls._lock:
            if cls._initialized:
                return
            cls.session = K.get_session()
            cls.graph = tf.get_default_graph()

            cls.__detector = RetinanetPinDetector(retinanet_pins_weights, warmup=True)
            # cls.__detector = PickledDictionaryPinDetector('detection/csv_cache/data/detections_video6.pcl')
            cls.__segmentation = UnetSceneSegmentation(unet_pins_weights, warmup=True)
            # cls.__segmentation = CachedSceneSegmentation('/home/trevol/HDD_DATA/Computer_Vision_Task/frames_6/not_augmented_base_vgg16_more_images_25')

            cls.graph.finalize()
            cls._initializer = cls._done
            cls._initialized = True

    _initializer = _create
    _initialized = False

    @classmethod
    def _done(cls): pass

    def __enter__(self):
        with self.session.as_default():
            with self.graph.as_default():
                return self

    def __exit__(self, *_): pass


if __name__ == '__main__':
    with ModelsContext() as ctx:
        assert ctx.detector is not None
    assert ModelsContext.detector is not None
