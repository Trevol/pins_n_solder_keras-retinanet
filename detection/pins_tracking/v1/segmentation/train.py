import argparse
import os

import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator

from detection.pins_tracking.v1.segmentation import LoadBatches
from detection.pins_tracking.v1.segmentation.classesMeta import BGR
from detection.pins_tracking.v1.segmentation.pin_utils import remainderlessDividable, colorizeLabel
from detection.pins_tracking.v1.segmentation.MyVGGUnet import VGGUnet


def base_train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    train_images_path = "dataset/image/"
    train_segs_path = "dataset/multi_class_masks/"
    train_batch_size = args.batch_size
    n_classes = 6
    input_height = remainderlessDividable(1080 // 2, 32, 1)
    input_width = remainderlessDividable(1920 // 2, 32, 1)

    save_weights_path = 'checkpoints/not_augmented_base_vgg16_more_images'

    vgg16NoTopWeights = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # vgg16NoTopWeights = None
    model = VGGUnet(n_classes, input_height=input_height, input_width=input_width,
                    vgg16NoTopWeights=vgg16NoTopWeights)
    model.load_weights('checkpoints/not_augmented_base_vgg16_more_images/unet_pins_16_0.000024_1.000000.hdf5')

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adadelta(),
                  metrics=['accuracy'])

    output_height = model.outputHeight
    output_width = model.outputWidth
    # print("Model output shape", model.output_shape, (output_height, output_width), (input_height, input_width))

    G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, train_batch_size, n_classes,
                                               input_height, input_width, output_height, output_width)
    os.makedirs(save_weights_path, exist_ok=True)

    chckPtsPath = os.path.join(save_weights_path, 'unet_pins_{epoch}_{loss:.6f}_{accuracy:.6f}.hdf5')
    model_checkpoint = ModelCheckpoint(chckPtsPath, monitor='loss', verbose=1, save_best_only=False,
                                       save_weights_only=True)
    model.fit_generator(G, steps_per_epoch=3000, epochs=25, callbacks=[model_checkpoint], initial_epoch=16)


class AugmentedTrainer:
    def __init__(self):
        pass

    def prepareDataForModel(self, imgBatch, maskBatch, nClasses, maskSize):
        # imgBatch = imgBatch / 255.0

        for img in imgBatch:
            img[2, :, :] -= 103.939
            img[1, :, :] -= 116.779
            img[0, :, :] -= 123.68

        labelsBatch = []
        height, width = maskSize
        for mask in maskBatch:
            labels = np.zeros((height, width, nClasses))
            mask = cv2.resize(mask, (width, height))
            for c in range(nClasses):
                labels[:, :, c] = (mask == c).astype(int)
            labels = np.reshape(labels, (width * height, nClasses))
            labelsBatch.append(labels)
        labelsBatch = np.array(labelsBatch, dtype=np.int32)

        return imgBatch, labelsBatch

    def normalizeRgbImage(self, img):
        assert img.shape[0] == 3
        normalized = np.empty_like(img)
        normalized[0, :, :] = img[0, :, :] - 123.68
        normalized[1, :, :] = img[1, :, :] - 116.779
        normalized[2, :, :] = img[2, :, :] - 103.939
        return normalized

    def maskToLabel(self, mask, nClasses, height, width):
        labels = np.zeros((height, width, nClasses))
        mask = cv2.resize(mask, (width, height))
        for c in range(nClasses):
            labels[:, :, c] = (mask == c).astype(int)
        labels = np.reshape(labels, (width * height, nClasses))
        return labels

    def batchMaskToLabel(self, maskBatch, nClasses, maskSize):
        labelsBatch = []
        height, width = maskSize
        for mask in maskBatch:
            labels = self.maskToLabel(mask, nClasses, height, width)
            labelsBatch.append(labels)
        labelsBatch = np.array(labelsBatch, dtype=np.int32)
        return labelsBatch

    def trainGenerator(self, batchSize, nClasses, trainFolder, imageFolder, maskFolder, imageSize, maskSize,
                       yieldOriginalBatches=False):
        aug_dict = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='constant',  # 'nearest'
                        cval=0
                        )
        aug_dict = {}
        seed = 1
        image_datagen = ImageDataGenerator(**aug_dict,
                                           data_format='channels_first',
                                           preprocessing_function=self.normalizeRgbImage
                                           )
        mask_datagen = ImageDataGenerator(**aug_dict)
        image_generator = image_datagen.flow_from_directory(
            trainFolder,
            classes=[imageFolder],
            class_mode=None,
            color_mode='rgb',
            target_size=imageSize,
            batch_size=batchSize,
            seed=seed)

        mask_generator = mask_datagen.flow_from_directory(
            trainFolder,
            classes=[maskFolder],
            class_mode=None,
            color_mode='grayscale',
            target_size=imageSize,
            batch_size=batchSize,
            interpolation='nearest',
            seed=seed)

        train_generator = zip(image_generator, mask_generator)

        for imgBatch, maskBatch in train_generator:
            labelBatch = self.batchMaskToLabel(maskBatch, nClasses, maskSize)
            yield imgBatch, labelBatch

    def train(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", type=int, default=2)
        args = parser.parse_args()

        trainPath = 'dataset'
        imageFolder = "image"
        maskFolder = "multi_class_masks"
        batchSize = args.batch_size
        n_classes = 6
        input_height = remainderlessDividable(1080 // 2, 32, 1)
        input_width = remainderlessDividable(1920 // 2, 32, 1)

        save_weights_path = 'checkpoints/not_augmented'

        model = MyVGGUnet.VGGUnet(n_classes, input_height=input_height, input_width=input_width,
                                  vgg16NoTopWeights='../../data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
                                  )
        # model.load_weights('checkpoints/augmented/1/unet_pins_augm_3_0.0062_0.9916.hdf5')

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adadelta(),
                      metrics=['accuracy'])

        output_height = model.outputHeight
        output_width = model.outputWidth
        maskSize = (output_height, output_width)

        # maskSize = 272, 496
        # print("Model output shape", model.output_shape, (output_height, output_width), (input_height, input_width))

        gen = self.trainGenerator(batchSize=batchSize, nClasses=n_classes, trainFolder=trainPath,
                                  imageFolder=imageFolder,
                                  maskFolder=maskFolder, imageSize=(input_height, input_width),
                                  maskSize=maskSize, yieldOriginalBatches=False)

        os.makedirs(save_weights_path, exist_ok=True)

        chckPtsPath = os.path.join(save_weights_path, 'unet_pins_augm_{epoch}_{loss:.5f}_{accuracy:.5f}.hdf5')
        model_checkpoint = ModelCheckpoint(chckPtsPath, monitor='loss', verbose=1, save_best_only=False,
                                           save_weights_only=True)
        model.fit_generator(gen, steps_per_epoch=3000, epochs=20, callbacks=[model_checkpoint])

    def vis(self, imgBatch, maskBatch, maskSize):
        for img, mask in zip(imgBatch, maskBatch):
            img = np.moveaxis(img, 0, 2)  # channels_first->channels_last
            mask = mask[..., 0]
            resizedMask = cv2.resize(mask, maskSize[::-1], interpolation=cv2.INTER_AREA)
            print(maskSize)
            cv2.imshow('img', img.astype(np.uint8)[..., ::-1])  # RGB to BGR
            cv2.imshow('mask', colorizeLabel(mask.astype(np.uint8), BGR))
            cv2.imshow('mask', colorizeLabel(resizedMask.astype(np.uint8), BGR))
            if cv2.waitKey() == 27:
                return False
        return True


def main():
    # AugmentedTrainer().train()
    base_train()


if __name__ == '__main__':
    main()
