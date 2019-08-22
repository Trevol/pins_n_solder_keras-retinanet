import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    f1 = plt.figure()

    f1.canvas.manager.window.showMaximized()
    f1.canvas.mpl_connect('resize_event', lambda e: f1.tight_layout())

    ax1, ax2 = f1.subplots(2, 1, sharex=True, sharey=False)
    ax1.set_title('AX_1')

    line11: Line2D = ax1.add_line(Line2D([1, 2], [1, 2]))
    ax1.set_ylim(0, 3)
    ax1.set_xlim(0, 3)
    line11.set_label('line11')

    ax2.set_title('AX_2')

    # f2 = plt.figure()
    plt.show()


if __name__ == '__main__':
    main()
