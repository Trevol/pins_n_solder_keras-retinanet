import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    f1 = plt.figure()

    f1.canvas.manager.window.showMaximized()
    f1.canvas.mpl_connect('resize_event', lambda e: f1.tight_layout())

    f1.subplots(2, 1, sharex=True, sharey=False)

    f2 = plt.figure()
    plt.show()


if __name__ == '__main__':
    main()
