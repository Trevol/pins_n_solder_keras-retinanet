import os

import glob
import cv2
import numpy as np

from segmentation.train import LoadBatches
from segmentation.classesMeta import BGR, classNames
from utils import colorizeLabel, putLegend
from utils import remainderlessDividable


def showLegend():
    img = np.zeros([300, 400, 3])
    putLegend(img, classNames, BGR)
    cv2.imshow('Legend', img)


def read_predict_show():
    from segmentation.MyVGGUnet import VGGUnet
    from collections import namedtuple
    FramesDesc = namedtuple('FramesDesc', 'imagesPath resultsPath height width')

    framesConfig = [
        FramesDesc(imagesPath='/HDD_DATA/Computer_Vision_Task/frames_6/',
                   resultsPath='/HDD_DATA/Computer_Vision_Task/frames_6/not_augmented_base_vgg16_more_images_25/',
                   height=1080 // 2,
                   width=1920 // 2),
        # FramesDesc(imagesPath='/home/trevol/HDD_DATA/Computer_Vision_Task/Computer_Vision_Task/frames_2/',
        #            resultsPath='/home/trevol/HDD_DATA/Computer_Vision_Task/Computer_Vision_Task/frames_2/unet_multiclass_no_augm_base/',
        #            height=1080 // 2,
        #            width=1920 // 2)
    ]

    weights = 'checkpoints/not_augmented_base_vgg16_more_images/unet_pins_25_0.000016_1.000000.hdf5'
    n_classes = 6

    for images_path, resultsPath, input_height, input_width in framesConfig:
        input_height = remainderlessDividable(input_height, 32, 1)
        input_width = remainderlessDividable(input_width, 32, 1)
        os.makedirs(resultsPath, exist_ok=True)

        showLegend()

        model = VGGUnet(n_classes, input_height=input_height, input_width=input_width)
        model.load_weights(weights)

        output_height = model.outputHeight
        output_width = model.outputWidth

        images = glob.glob(images_path + "*.jpg")
        images.sort()

        # annotations_path = 'dataset/multi_class_masks/'
        annotations_path = None

        for imgName in images:

            # X = np.float32(cv2.imread(imgName)) / 255
            # X = cv2.resize(X, (input_width, input_height))
            # X = np.moveaxis(X, 2, 0)  # channel_first
            X = LoadBatches.getImageArr(imgName, input_width, input_height, imgNorm="sub_mean",
                                        ordering='channels_first',
                                        as_rgb=False)

            pr = model.predict(np.expand_dims(X, 0))[0]
            probabilities = pr.reshape((output_height, output_width, n_classes))
            labelsImage = probabilities.argmax(axis=2)

            seg_img = colorizeLabel(labelsImage, BGR)
            # seg_img = cv2.resize(seg_img, (input_width, input_height))

            input = cv2.imread(imgName)

            # gtPath = None if annotations_path is None \
            #     else os.path.join(annotations_path, os.path.basename(imgName).replace('.jpg', '.png'))
            # if gtPath is not None and os.path.isfile(gtPath):
            #     gt = cv2.imread(gtPath, cv2.IMREAD_GRAYSCALE)
            #     gt_colored = colorizeLabel(gt, BGR)
            #     cv2.imshow('gt', gt_colored)
            # else:
            #     cv2.imshow('gt', np.uint8([[0]]))

            cv2.imshow('input', input)
            cv2.imshow('output', seg_img)

            # outName = imgName.replace(images_path, resultsPath).replace('.jpg', '.png')
            # cv2.imwrite(outName, seg_img)
            # outName = imgName.replace(images_path, resultsPath).replace('.jpg', '_label.png')
            # cv2.imwrite(outName, np.uint8(labelsImage))

            # outName = imgName.replace(images_path, resultsPath).replace('.jpg', '.npy')
            # np.save(outName, probabilities)

            if cv2.waitKey(1) == 27:
                break
        ##################################
        cv2.waitKey()

    cv2.destroyAllWindows()


def view_base_train_results():
    root = '/HDD_DATA/Computer_Vision_Task/frames_6'
    paths = [
        os.path.join(root, '*.jpg'),
        os.path.join(root, 'not_augmented_base_vgg16_more_images_20', '*.png'),
        # os.path.join(root, 'unet_multiclass_no_augm_base', '*.png'),
        # os.path.join(root, 'unet_multiclass_no_augm_base_13', '*.png'),
        # os.path.join(root, 'unet_multiclass_no_augm_base_15', '*.png'),
        # os.path.join(root, 'unet_multiclass_no_augm_base_18', '*.png'),
        # os.path.join(root, 'unet_multiclass_no_augm_base_20', '*.png')
    ]

    # classProbabilitesPath = os.path.join(root, 'unet_multiclass_no_augm_base_20')
    # classProbabilites = None

    # init windows
    # def showPointClassProbabilities(evt, x, y, flags, param):
    #     if evt == cv2.EVENT_LBUTTONDOWN:
    #         print(classProbabilites[y, x])

    for i in range(len(paths)):
        windowName = str(i)
        cv2.namedWindow(windowName)
        # cv2.setMouseCallback(windowName, showPointClassProbabilities)

    imagesPaths = [sorted(glob.glob(p)) for p in paths]
    imagesCount = len(imagesPaths[0])
    for i in range(0, imagesCount):
        pathsOfImageI = [images[i] for images in imagesPaths]

        # load classProbabilities
        # probaPath = os.path.join(classProbabilitesPath, os.path.basename(pathsOfImageI[0]).replace('.jpg', '.npy'))
        # classProbabilites = np.load(probaPath)
        # print(classProbabilites.shape, classProbabilites.dtype, classProbabilites.min(), classProbabilites.max())

        for windowInd, path in enumerate(pathsOfImageI):
            image = cv2.imread(path)
            # image = cv2.resize(image, (496, 272))
            windowName = str(windowInd)
            cv2.imshow(windowName, image)
            parentDir, fileName = path.split('/')[-2:]
            parentDirId = parentDir.split('_')[-1]
            cv2.setWindowTitle(windowName, f"{parentDirId}   {fileName}")

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


def main():
    read_predict_show()
    # view_base_train_results()


main()
