import cv2
import numpy as np
import os
import csv


def readColorMap(colorMapFile):
    with open(colorMapFile, 'rt', newline='') as txt:
        for labelId, (labelName, colorStr) in enumerate(csv.reader(txt, delimiter=':')):
            rgbColor = np.fromstring(colorStr, np.uint8, sep=',')
            bgrColor = rgbColor[[2, 1, 0]]
            yield labelId, labelName, rgbColor, bgrColor


def colorLabel2labelIdImage(colorLabelImage, colorMap):
    labelIdImage = np.zeros(colorLabelImage.shape[:2], np.uint8)
    for labelId, labelName, colorRgb, colorBgr in colorMap:
        mask = cv2.inRange(colorLabelImage, colorBgr, colorBgr)
        np.place(labelIdImage, mask, labelId)
    return labelIdImage


def enumerateColorMapsPngs(dir):
    _, _, filenames = next(os.walk(dir))
    return [file for file in filenames if file.endswith('.png') and not file.endswith('_mask.png')]


def main():
    labelDir = 'label'
    colorMap = list(readColorMap(f'{labelDir}/colormap.txt'))
    # pinOnlyColorMap = [
    #     (0, 'background', np.uint8([0, 0, 0]), np.uint8([0, 0, 0])),
    #     (255, 'pin', np.uint8([128, 0, 0]), np.uint8([0, 0, 128]))
    # ]
    # solderOnlyColorMap = [
    #     (0, 'background', np.uint8([0, 0, 0]), np.uint8([0, 0, 0])),
    #     (255, 'pin_w_solder', np.uint8([0, 128, 0]), np.uint8([0, 128, 0]))
    # ]
    # colorMaps = enumerateColorMapsPngs(labelDir)
    colorMaps = ['f_3439_229266.67_229.27.png',
                 'f_3975_265000.00_265.00.png',
                 'f_4446_296400.00_296.40.png',
                 'f_4765_317666.67_317.67.png']
    for pngFile in colorMaps:
        print(pngFile)
        colorLabelsImage = cv2.imread(os.path.join(labelDir, pngFile))

        idImage = colorLabel2labelIdImage(colorLabelsImage, colorMap)
        maskFile = 'multi_class_masks/' + pngFile
        cv2.imwrite(maskFile, idImage)

        # # pin-only
        # idImage = colorLabel2labelIdImage(colorLabelsImage, pinOnlyColorMap)
        # maskFile = 'pin_only_masks/' + pngFile
        # cv2.imwrite(maskFile, idImage)
        #
        # # solder-only
        # idImage = colorLabel2labelIdImage(colorLabelsImage, solderOnlyColorMap)
        # maskFile = 'solder_only_masks/' + pngFile
        # cv2.imwrite(maskFile, idImage)


main()
