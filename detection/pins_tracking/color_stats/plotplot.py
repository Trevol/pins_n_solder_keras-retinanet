import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot():
    def data_gen(t=0):
        cnt = 0
        while cnt < 1000:
            cnt += 1
            t += 0.1
            yield t, np.sin(2 * np.pi * t) * np.exp(-t / 10.)

    def init():
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlim(0, 10)
        del xdata[:]
        del ydata[:]
        line.set_data(xdata, ydata)
        return line,

    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=1)
    ax.grid()
    xdata, ydata = [], []

    def run(data):
        # update the data
        t, y = data
        xdata.append(t)
        ydata.append(y)
        xmin, xmax = ax.get_xlim()

        if t >= xmax:
            ax.set_xlim(xmin, 2 * xmax)
            ax.figure.canvas.draw()
        line.set_data(xdata, ydata)

        return line,

    ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=10,
                                  repeat=False, init_func=init)
    plt.show(block=True)


class Plotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], lw=1)
        self.ax.grid()
        self.xdata, self.ydata = [], []

    def data_gen(self, t=0):
        cnt = 0
        while cnt < 1000:
            cnt += 1
            t += 0.1
            yield t, np.sin(2 * np.pi * t) * np.exp(-t / 10.)

    def initAnimation(self):
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_xlim(0, 10)
        del self.xdata[:]
        del self.ydata[:]
        self.line.set_data(self.xdata, self.ydata)
        return self.line,

    def run(self, data):
        # update the data
        t, y = data
        self.xdata.append(t)
        self.ydata.append(y)
        xmin, xmax = self.ax.get_xlim()

        if t >= xmax:
            self.ax.set_xlim(xmin, 2 * xmax)
            self.ax.figure.canvas.draw()
        self.line.set_data(self.xdata, self.ydata)

        return self.line,

    def plot(self):
        ani = animation.FuncAnimation(self.fig, self.run, self.data_gen, blit=False, interval=10,
                                      repeat=False, init_func=self.initAnimation)
        plt.show(block=True)


def plotWithPlotter():
    plotter = Plotter()
    plotter.plot()


if __name__ == '__main__':
    plotWithPlotter()
