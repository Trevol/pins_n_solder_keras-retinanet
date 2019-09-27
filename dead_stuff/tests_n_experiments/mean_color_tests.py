import numpy as np


def stats(colors):
    axis = 0
    return np.mean(colors, axis), np.std(colors, axis), np.max(colors, axis), np.min(colors, axis)

def main():
    # lastColor = [174.7182, 187., 229.137]
    meanColors = np.load('./meanColors_2_higher_contrast_700_247.npy')
    lastPinColorIndex = 638
    pinColors = meanColors[:lastPinColorIndex + 1]
    solderColors = meanColors[lastPinColorIndex + 1:]


    print(*stats(pinColors))
    print(*stats(solderColors))


main()
