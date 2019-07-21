class InstanceTracker:
    def __init__(self):
        self.__instances = []

    def draw(self, img):
        for instance in self.__instances:
            instance.draw()

    def findInstance(self, bbox):
        return self.__instances[-1] if any(self.__instances) else None
        raise NotImplementedError()
        pass

    def applyRawFrameDetections(self, rawDetections, framePos, frame):
        # признак стабильной сцены:
        # - N последовательных кадров определяется одинаковое кол-во объектов (list of bbox) на неизменных позициях
        #
        # ddd
        for bbox, label, score in rawDetections:
            instance = self.findInstance(bbox)
            if instance:
                instance.addBBox(bbox)
            else:
                instance = DetectedInstance(bbox)
                self.__instances.append(instance)
                # TODO: remove (hide and remove in next frame processing)


class DetectedInstance:
    def __init__(self, bbox):
        self.__bboxes = [bbox]

    def addBBox(self, bbox):
        self.__bboxes.append(bbox)

    def draw(self):
        pass
