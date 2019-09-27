import cv2
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from utils import resize
from utils import VideoPlayback
from utils.VideoPlaybackHandlerBase import VideoPlaybackHandlerBase


def uniqueItems(items):
    uniq = []
    for item in items:
        itemIsUnique = True
        for seenItem in uniq:
            if np.array_equal(seenItem, item):
                itemIsUnique = False
                break
        if itemIsUnique:
            uniq.append(item)
    return uniq


def BGR2HSV(bgr):
    bgr = np.reshape(bgr, (bgr.shape[0], 1, 3))
    hsv = cv2.cvtColor(np.uint8(bgr), cv2.COLOR_BGR2HSV)
    hsv = np.reshape(hsv, (hsv.shape[0], 3))

    return hsv


def trainTree(trainData, trainLabels):
    clf = tree.DecisionTreeClassifier(criterion='entropy')

    clf = clf.fit(trainData, trainLabels)

    # print(trainData.shape)
    # print(trainLabels.shape)
    # print(testData.shape)
    # print(testLabels.shape)
    #
    # print(clf.feature_importances_)
    # print(clf.score(testData, testLabels))

    return clf

def trainSVM(trainData, trainLabels):
    # clf = svm.SVC(gamma='scale')
    # clf = linear_model.SGDClassifier()
    clf = LinearSVC()
    clf.fit(trainData, trainLabels)
    return clf

def applyToImage(img, classifier):
    data = np.reshape(img, (img.shape[0] * img.shape[1], 3))
    predictedLabels = classifier.predict(data)
    # imgLabels = np.reshape(predictedLabels, (img.shape[0], img.shape[1], 1))
    imgLabels = np.reshape(predictedLabels, (img.shape[0], img.shape[1]))
    return imgLabels.astype(np.uint8)


def files():
    yield '/HDD_DATA/Computer_Vision_Task/Video_6.mp4'
    # yield '/HDD_DATA/Computer_Vision_Task/Video_2.mp4'


class VideoHandler(VideoPlaybackHandlerBase):
    def __init__(self, frameSize, colorClassifier):
        super(VideoHandler, self).__init__(frameSize)
        self._frameScaleFactor = .85
        self.colorClassifier = colorClassifier
        self.lut = np.zeros([1, 256, 3], np.uint8)
        self.lut[0, 1:3] = [
            [0, 0, 200],  # 1- pin - red
            [255, 255, 255],  # 2 - pin_solder - white
        ]

    def labelsToDisplay(self, labelsImg):
        labelsWith3Channels = np.dstack([labelsImg, labelsImg, labelsImg])
        disp = cv2.LUT(labelsWith3Channels, self.lut)
        return disp

    def frameReady(self, frame, framePos, framePosMsec, playback):
        super(VideoHandler, self).frameReady(frame, framePos, framePosMsec, playback)
        frame = resize(frame, 0.5)
        # TODO: try different classifiers
        lblImage = applyToImage(frame, self.colorClassifier)
        lutImage = self.labelsToDisplay(lblImage)
        cv2.imshow('lutImage', lutImage)


def main():
    data = np.genfromtxt('dataset.txt', dtype=np.int32)
    bgrData = data[:, 0:3]
    labels = data[:, 3]

    bgrTrainData, bgrTestData, bgrTrainLabels, bgrTestLabels = train_test_split(bgrData, labels, test_size=0.20,
                                                                                random_state=42)
    # clf = trainTree(bgrTrainData, bgrTrainLabels)
    clf = trainSVM(bgrTrainData, bgrTrainLabels)

    ################################################################
    for sourceVideoFile in files():
        videoPlayback = VideoPlayback(sourceVideoFile, 1, autoplayInitially=False)
        handler = VideoHandler(videoPlayback.frameSize(), clf)

        # framesRange = (4150, None)
        framesRange = None
        videoPlayback.play(range=framesRange, onFrameReady=handler.frameReady, onStateChange=handler.syncPlaybackState)
        cv2.waitKey()
        videoPlayback.release()
        handler.release()

    cv2.waitKey()


main()
