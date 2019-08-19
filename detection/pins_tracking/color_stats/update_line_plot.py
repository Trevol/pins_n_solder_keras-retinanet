import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    # plt.ion()
    fig, ax = plt.subplots()
    ax.set_ylim(0, 10)
    ax.set_xlim(0, 10)

    line = ax.add_line(Line2D([], [], markersize='1', marker='o', linestyle=''))

    # line.set_color('g')
    # ax.autoscale_view(scalex=True, scaley=True)
    # ax.plot([1, 2, 3], [7, 8, 9])
    # ax.draw_artist(line)
    # plt.draw()
    # plt.pause(2)
    # plt.show()
    # plt.pause(3)
    line.set_data([2, 3, 5], [7, 8, 2])
    plt.pause(2)


if __name__ == '__main__':
    main()
