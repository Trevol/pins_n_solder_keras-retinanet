import cv2

from utils import VideoPlayback


def readFrame():
    pb = VideoPlayback('/HDD_DATA/Computer_Vision_Task/Video_6.mp4')
    frame = pb.readFrame(4651)
    pb.release()
    return frame


class Annotator:
    wnd = 'Annotator'

    def __init__(self, img, labelsIdNameMap):
        assert any(labelsIdNameMap)
        self.img = img
        self.labelsIdNameMap = labelsIdNameMap
        self.labeledData = {lblId: [] for lblId in self.labelsIdNameMap}
        self.currentLabelIdName = None
        self.currentLabelData = None
        self.currentImgWithPoints = None

    def startAnnotation(self):
        cv2.namedWindow(self.wnd)
        cv2.setMouseCallback(self.wnd, self.mouseHandler, None)
        for labelId, labelName in self.labelsIdNameMap.items():
            goToNextSession = self.annotationSession(labelId, labelName)
            if not goToNextSession:
                break

    def mouseHandler(self, evt, x, y, flags, param):
        if evt == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            self.addPoint(x, y)
        elif evt == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY:
            self.addPoint(x, y)

    def addPoint(self, x, y):
        dataItem = (self.img[y, x], x, y)
        self.currentLabelData.append(dataItem)
        self.newPointAdded(x, y)

    def newPointAdded(self, x, y):
        self.putPoint(self.currentImgWithPoints, x, y)
        cv2.imshow(self.wnd, self.currentImgWithPoints)

    def annotationSession(self, labelId, labelName):
        self.currentLabelIdName = labelId, labelName
        self.currentLabelData = self.labeledData[labelId]
        self.currentImgWithPoints = self.img.copy()

        cv2.imshow(self.wnd, self.currentImgWithPoints)
        cv2.setWindowTitle(self.wnd, f'Annotating {self.currentLabelIdName}')
        try:
            while True:
                key = cv2.waitKey()
                if key == 27:
                    return False  # exit app
                if key == ord('n'):
                    return True  # query session for next label
                elif key == ord('s'):
                    self.saveData()
                elif key == ord('c'):
                    self.undoLastPoint()
        finally:
            # cleanup
            self.currentLabelIdName = None
            self.currentLabelData = None
            self.currentImgWithPoints = None
            cv2.setWindowTitle(self.wnd, self.wnd)

    def saveData(self):
        with open('dataset.txt', 'w') as file:
            lines = (f'{b} {g} {r} {labelId}' for labelId, data in self.labeledData.items() for (b, g, r), x, y in data)
            for i, line in enumerate(lines):
                if i > 0:
                    file.write('\n')
                file.write(line)
            print('saved!')

    def undoLastPoint(self):
        if len(self.currentLabelData):
            del self.currentLabelData[-1]
            self.currentImgWithPoints = self.img.copy()
            for _, x, y in self.currentLabelData:
                self.putPoint(self.currentImgWithPoints, x, y)
            cv2.imshow(self.wnd, self.currentImgWithPoints)

    @staticmethod
    def putPoint(img, x, y, color=(0, 0, 200)):
        img[y - 1:y + 2, x - 1: x + 1] = color


def main():
    labelsIdNameMap = {1: 'Pin', 2: 'Pin_Solder', 3: 'Other'}
    frame = readFrame()
    frame = frame[70:-70]
    a = Annotator(frame, labelsIdNameMap)
    a.startAnnotation()


main()
