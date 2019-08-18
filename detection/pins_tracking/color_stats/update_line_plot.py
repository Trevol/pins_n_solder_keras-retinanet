import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    # plt.ion()
    fig, ax = plt.subplots()
    ax.set_ylim(0, 10)
    ax.set_xlim(0, 10)

    line = Line2D([1, 2], [3, 4])
    ax.add_line(line)
    # ax.autoscale_view(scalex=True, scaley=True)

    def onClick(event):
        # line.set_xdata([1, 2])
        line.set_ydata([2, 3])

        # ax.autoscale_view(scalex=True, scaley=True)
        fig.canvas.draw()

    # fig.canvas.mpl_connect('button_press_event', onClick)



    # line.set_color('g')
    # ax.autoscale_view(scalex=True, scaley=True)
    # ax.plot([1, 2, 3], [7, 8, 9])
    # ax.draw_artist(line)
    # plt.draw()
    # plt.pause(2)
    # plt.show()
    plt.pause(3)
    line.set_data([2, 3, 5], [7, 8, 2])
    plt.pause(3)


if __name__ == '__main__':
    main()
