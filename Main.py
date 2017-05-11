import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from lib import for_kin, plotlib, mathlib

def render_gst(gst):
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    print(gst)
    for i in range(1, len(gst)):
        plotlib.create_line(ax, mathlib.get_position_from_matrix(gst[i-1]), mathlib.get_position_from_matrix(gst[i]))
        plotlib.create_axis(ax, mathlib.get_position_from_matrix(gst[i]), mathlib.Quaternion.from_matrix(gst[i]))

    ax.legend()

    plt.show()


if __name__ == "__main__":
    theta = [0, 0, 0, 0, 0, 0, 0]
    Gst = for_kin.GST(theta)

    render_gst(Gst)