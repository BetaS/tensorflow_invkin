
def create_axis(axis, pos, ori):
    create_line(axis, pos, pos+ori.distance(0.1, 0, 0), color=(1, 0, 0), size=[0.01, 0.02, 0.02])
    create_line(axis, pos, pos+ori.distance(0, 0.1, 0), color=(0, 1, 0), size=[0.01, 0.02, 0.02])
    create_line(axis, pos, pos+ori.distance(0, 0, 0.1), color=(0, 0, 1), size=[0.01, 0.02, 0.02])


def create_line(axis, start, end, color=(0.2, 0.2, 0.2), size=[0.02, 0.05, 0.05]):
    axis.plot(xs=[start[0], end[0]], ys=[start[1], end[1]], zs=[start[2], end[2]], color=color)