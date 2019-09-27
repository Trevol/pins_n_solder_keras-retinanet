import glob
import numpy as np
import cv2
import random
import argparse

from src_trevol.pins.classesMeta import BGR
from src_trevol.pins.pin_utils import colorizeLabel


def imageSegmentationGenerator(images_path, segs_path, n_classes, colors=None):
    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
    images.sort()
    segmentations = glob.glob(segs_path + "*.jpg") + glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg")
    segmentations.sort()

    colors = colors or [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in
                        range(n_classes)]

    assert len(images) == len(segmentations)

    for im_fn, seg_fn in zip(images, segmentations):
        assert im_fn.split('/')[-1].split('.')[0] == seg_fn.split('/')[-1].split('.')[0]

        img = cv2.imread(im_fn)
        seg = cv2.imread(seg_fn)
        seg_img = colorizeLabel(seg[:, :, 0], colors)

        cv2.imshow("img", img)
        cv2.imshow("seg_img", seg_img)
        if cv2.waitKey() == 27:
            break


def main():
    imagesDir = 'dataset/image/'
    annotationsDir = 'dataset/multi_class_masks/'
    imageSegmentationGenerator(imagesDir, annotationsDir, classes=len(BGR), colors=BGR)


main()
