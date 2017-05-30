import matplotlib as mpl
mpl.use('Qt5Agg')

from pprint import pprint
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from lib import for_kin, plotlib, mathlib
import time
import math
from matplotlib.widgets import Slider, Button
import matplotlib.backends.backend_tkagg as tkagg
from lib.const import joint_name, joint_limit


def zoom_factory(ax, base_scale = 2.):
    def zoom_fun(event):
        # get the current x and y limits

        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_zlim = ax.get_zlim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        cur_zrange = (cur_zlim[1] - cur_zlim[0])*.5
        xdata = 0#event.xdata # get event x location
        ydata = 0#event.ydata # get event y location
        zdata = 0#event.zdata # get event z location

        scale_factor = 1
        if event.key == ']':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.key == '[':
            # deal with zoom out
            scale_factor = base_scale

        # set new limits
        ax.set_xlim([xdata - cur_xrange*scale_factor,
                     xdata + cur_xrange*scale_factor])
        ax.set_ylim([ydata - cur_yrange*scale_factor,
                     ydata + cur_yrange*scale_factor])
        ax.set_zlim([zdata - cur_zrange*scale_factor,
                     zdata + cur_zrange*scale_factor])
        plt.draw() # force re-draw

    fig = ax.get_figure() # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('key_press_event',zoom_fun)

    #return the function
    return zoom_fun


def pan_factory(ax, dist=.25):
    def pan_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        """
        
        cur_xrange = (math.fabs(cur_xlim[1]) + math.fabs(cur_xlim[0])) / 2 * dist
        cur_yrange = (math.fabs(cur_ylim[1]) + math.fabs(cur_ylim[0])) / 2 * dist
        
        """
        cur_xrange = (cur_xlim[1] - cur_xlim[0]) * dist
        cur_yrange = (cur_ylim[1] - cur_ylim[0]) * dist

        xdata = 0
        ydata = 0

        azi = math.radians(ax.azim+90)

        if event.key == 'up':
            xdata = math.sin(azi)
            ydata = math.cos(azi)
        elif event.key == 'down':
            xdata = -math.sin(azi)
            ydata = -math.cos(azi)
        elif event.key == 'left':
            xdata = math.cos(azi)
            ydata = math.sin(azi)
        elif event.key == 'right':
            xdata = -math.cos(azi)
            ydata = -math.sin(azi)

        xdata *= cur_xrange
        ydata *= cur_yrange

        print(azi, xdata, ydata)

        print(cur_xlim)
        # set new limits
        ax.set_xlim([cur_xlim[0] + xdata,
                     cur_xlim[1] + xdata])
        ax.set_ylim([cur_ylim[0] + ydata,
                     cur_ylim[1] + ydata])

        plt.draw()  # force re-draw

    fig = ax.get_figure()  # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('key_press_event', pan_fun)

    # return the function
    return pan_fun


def init_matplot():
    fig = plt.figure(figsize=[9, 5])
    fig.subplots_adjust(left=0, right=0.5, top=1, bottom=0.25)
    ax = fig.gca(projection='3d')
    ax.set_xlabel("X")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylabel("Y")
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlabel("Z")
    ax.set_zlim(-0.1, 1.1)

    scale = 1.5
    zoom_factory(ax, base_scale=scale)
    pan_factory(ax)

    axcolor = 'lightgoldenrodyellow'

    sliders = []

    def update(val):
        theta = []
        for i in range(7):
            theta.append(math.radians(sliders[i]["sl"].val))

        Gst = for_kin.GST(theta)
        render_gst(ax, Gst)

        #fig.canvas.draw_idle()


    for i in range(7):
        obj = {"ax": None, "sl": None}

        obj["ax"] = plt.axes([0.6, 0.9-(i*0.07), 0.3, 0.05], facecolor=axcolor)
        obj["sl"] = Slider(obj["ax"], joint_name[i], joint_limit[i][0], joint_limit[i][1], valinit=0, valfmt='%.02f\'', dragging=True)

        obj["sl"].on_changed(update)

        sliders.append(obj)

    return ax

def render_gst(ax, gst):
    #print(gst)

    #while True:
    #for line in ax.lines:
    #    line.remove()

    del ax.lines[:]
    for i in range(1, len(gst)):
        plotlib.create_line(ax, mathlib.get_position_from_matrix(gst[i - 1]),
                            mathlib.get_position_from_matrix(gst[i]))
        plotlib.create_axis(ax, mathlib.get_position_from_matrix(gst[i]), mathlib.Quaternion.from_matrix(gst[i]))



if __name__ == "__main__":
    #plt.ion()

    ax = init_matplot()

    theta = [0, 0, 0, 0, 0, 0, 0]
    Gst = for_kin.GST(theta)

    render_gst(ax, Gst)

    plt.show()
    """
    while True:
        plt.draw()
        plt.pause(0.01)
    """
