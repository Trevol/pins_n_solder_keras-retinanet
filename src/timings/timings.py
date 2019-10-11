import pickle

from matplotlib.lines import Line2D


class __Timings:
    data = []

    def newRecord(self, numOfValues=3, defaultValue=0):
        self.data.append([defaultValue] * numOfValues)

    def addValue(self, index, value):
        rec = self.data[-1]
        rec[index] = value

    def save(self, fileName='timings.pcl'):
        with open(fileName, 'wb') as file:
            pickle.dump(self.data, file, -1)

    def load(self, fileName='timings.pcl'):
        with open(fileName, 'rb') as file:
            return pickle.load(file)

    def loadAndPlot(self, fileName='timings.pcl'):
        import numpy as np

        data = self.load(fileName)
        totals = []
        detections = []
        segmentations = []
        for total, detection, segmentation in data:
            totals.append(total)
            detections.append(detection)
            segmentations.append(segmentation)
        print('total: ', np.mean(totals), np.median(totals), np.std(totals))
        print('detections: ', np.mean(detections), np.median(detections), np.std(detections))
        actualSegmentations = [s for s in segmentations if s > 0]
        print('segmentations: ', np.mean(actualSegmentations), np.median(actualSegmentations),
              np.std(actualSegmentations))

        # plot
        import matplotlib.pyplot as plt

        f = plt.figure()
        ax = f.subplots(1, 1)
        ax.set_xlim(0, len(totals))

        x = range(len(totals))
        ax.add_line(Line2D(x, totals, markersize='1', color='#808000'))  # olive
        ax.add_line(Line2D(x, detections, markersize='1', color='#808080'))  # gray
        ax.add_line(Line2D(x, segmentations, markersize='1', color='#800080'))  # Purple

        plt.show()


timings = __Timings()

if __name__ == '__main__':
    timings.loadAndPlot('timings_thread_0.pcl')


def test():
    for i in range(1, 4):
        timings.newRecord(3, 0)
        timings.addValue(0, 10.01 * i)
        timings.addValue(1, 20.01 * i)
        timings.addValue(2, 30.01 * i)

    timings.save('timings_test.pcl')
    data = timings.load('timings_test.pcl')
    print(data)
