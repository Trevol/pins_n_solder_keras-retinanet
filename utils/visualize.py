import numpy as np
import cv2
from keras_retinanet.utils.visualization import draw_box
from keras_retinanet.utils.colors import label_color
from . import boxCenter


def drawDetections(image, detections, drawCenters=False):  # visualize detections
    for box, label, score in detections:
        # color = label_color(label)
        if score < 0.96:
            color = (0, 0, 255)
        elif label == 1:  # solder
            color = (0, 255, 0)
        else:
            color = (200, 0, 0)
        b = np.round(box, 0).astype(int)
        draw_box(image, b, color=color, thickness=1)
        if drawCenters:
            center = boxCenter(box, roundPt=True)
            cv2.circle(image, tuple(center), 1, color)

        if score < .96:
            draw_caption(image, b, str(int(score * 100)), fontScale=0.7)
    return image


def drawBoxes(image, boxes, drawCenters=False, color=(200, 0, 0)):  # visualize detections
    for box in boxes:
        b = np.round(box, 0).astype(int)
        draw_box(image, b, color=color, thickness=1)
        if drawCenters:
            center = boxCenter(box, roundToInt=True)
            cv2.circle(image, tuple(center), 1, color)
    return image


def draw_caption(image, box, caption, fontScale=1):
    """ Draws a caption above the box in an image.

    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """

    cv2.putText(image, caption, (box[0], box[1] + 7), cv2.FONT_HERSHEY_PLAIN, fontScale, (0, 0, 0), 2)
    cv2.putText(image, caption, (box[0], box[1] + 7), cv2.FONT_HERSHEY_PLAIN, fontScale, (255, 255, 255), 1)


def putFramePos(point, frame, pos, posMsec=None):
    x, y = point
    cv2.putText(frame, str(pos), point, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
    if posMsec is not None:
        cv2.putText(frame, f'{posMsec:.1f}ms', (x, y + 30), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 0, 255))
    return frame


def colorizeLabel(label, colors):
    colorized = np.zeros(label.shape[:2] + (3,), np.uint8)
    for classId, color in enumerate(colors):
        classMask = label == classId
        colorized[..., 0] += np.multiply(classMask, color[0], dtype=np.uint8, casting='unsafe')
        colorized[..., 1] += np.multiply(classMask, color[1], dtype=np.uint8, casting='unsafe')
        colorized[..., 2] += np.multiply(classMask, color[2], dtype=np.uint8, casting='unsafe')
    return colorized


def putLegend(img, names, colors):
    imgHeight = img.shape[0]
    x = 40
    yBase = imgHeight - 40
    yStep = 40
    for i, (name, color) in enumerate(zip(names, colors)):
        y = yBase - i * yStep
        cv2.putText(img, name, (x, y), cv2.FONT_HERSHEY_COMPLEX, .5, color)