import numpy as np
import cv2
from keras_retinanet.utils.visualization import draw_box
from keras_retinanet.utils.colors import label_color


def drawDetections(image, detections):  # visualize detections
    for box, label, score in detections:
        # color = label_color(label)
        if score < 0.99:
            color = (0, 0, 255)
        elif label == 1:  # solder
            color = (0, 255, 0)
        else:
            color = (0, 255, 255)
        b = np.round(box, 0).astype(int)
        draw_box(image, b, color=color, thickness=1)

        # caption = f"{labels_to_names[label]} {score:.2f}"
        # draw_caption(draw, b, caption)
        if score < 1.0:
            draw_caption(image, b, str(int(score * 100)), fontScale=0.7)
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


def putFramePos(frame, pos):
    cv2.putText(frame, str(int(pos)), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
    return frame