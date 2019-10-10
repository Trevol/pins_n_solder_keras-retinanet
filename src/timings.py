import pickle


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
        data = self.load(fileName)
        print(len(data))
        print(len(data[-1]))


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
