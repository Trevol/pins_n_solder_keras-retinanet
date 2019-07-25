def pointOfInterest_6():
    x, y = 1073 / 0.7, 569 / 0.7  # cords taken from downsized image - so restore original coords
    return x, y


def pointOfInterest_2():
    x, y = 700, 247
    # x, y = 759, 246
    # x, y = 636, 350
    return x, y


meanColorBuffer = []

def saveMeanColorBuffer():
    # assert len(meanColorBuffer) > 20
    # import numpy as np
    # np.save('./meanColors_2_higher_contrast_700_247.npy', np.array(meanColorBuffer, dtype=np.float32))
    pass

# DEBUG
# boxOfIntereset = Box.boxByPoint(self.__meanBoxes, pointOfInterest_2())
# if boxOfIntereset is None:
#     print('boxOfInterest is None')
# else:
#     meanColor, colorStd = self.__boxOuterColorStats(frame, boxOfIntereset)
#     meanColorBuffer.append(meanColor)
