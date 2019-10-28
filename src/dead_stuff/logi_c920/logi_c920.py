import cv2
import numpy as np

from utils.Timer import timeit


def main():
    video = cv2.VideoCapture()
    if not video.open(0, cv2.CAP_ANY):
        print('Not opened!!')
        return
    print(video.getBackendName())

    video.set(cv2.CAP_PROP_CONVERT_RGB, False)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # fourcc = cv2.VideoWriter_fourcc('B', 'G', 'R', '3')
    # fourcc = cv2.VideoWriter_fourcc('R', 'G', 'B', '3')
    # fourcc = cv2.VideoWriter_fourcc('Y', 'U', '1', '2')
    # fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    video.set(cv2.CAP_PROP_FOURCC, fourcc)

    # w, h = 1280, 720
    w, h = 1600, 896
    # w, h = 1920, 1080
    video.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    video.set(cv2.CAP_PROP_FPS, 30)
    video.set(cv2.CAP_PROP_AUTOFOCUS, False)
    # video.set(cv2.CAP_PROP_AUTOFOCUS, True)

    video.set(cv2.CAP_PROP_AUTO_WB, False)

    print('----------------------')
    print(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(video.get(cv2.CAP_PROP_FPS))
    print(video.get(cv2.CAP_PROP_AUTOFOCUS))
    print(video.get(cv2.CAP_PROP_AUTO_WB))
    print(video.get(cv2.CAP_PROP_FOURCC))
    print(fourCCStringFromCode(int(video.get(cv2.CAP_PROP_FOURCC))))
    print(video.get(cv2.CAP_PROP_CONVERT_RGB))
    print(video.get(cv2.CAP_PROP_FOCUS))
    # return
    # frame = np.empty([h, w, 3], np.uint8)

    currentFocus = 35
    video.set(cv2.CAP_PROP_FOCUS, currentFocus)
    print('Focus:', video.get(cv2.CAP_PROP_FOCUS))
    while True:
        video.grab()
        ret, frame = video.retrieve()
        if not ret:
            print('not ret !!!!')
            continue
        frame = cv2.imdecode(frame, -1)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key in (ord('f'), ord('c')):
            currentFocus += 5 if key == ord('f') else -5
            video.set(cv2.CAP_PROP_FOCUS, currentFocus)
            print('Focus:', video.get(cv2.CAP_PROP_FOCUS), currentFocus)

    video.release()


# def fourCCStringFromCode(code):
#     for (int i = 0; i < 4; i++) {
#         fourCC[3 - i] = code >> (i * 8);
#     }
#     fourCC[4] = '\0';

def fourCCStringFromCode(code):
    byteMask = 255
    fourCC = [0] * 4
    for i in range(4):
        b = code >> (i * 8) & byteMask
        fourCC[i] = chr(b)
    return ''.join(fourCC)


main()
