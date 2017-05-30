import numpy as np
from lib import for_kin, plotlib, mathlib
import time
import math
from lib.const import joint_name, joint_limit
import random

def generate_tc():
    q = []
    for i in range(7):
        q.append(random.uniform(math.radians(joint_limit[i][0]), math.radians(joint_limit[i][1])))

    return q

if __name__ == "__main__":
    f = open("testset_red.txt", "w")
    for i in range(1000):
        theta = generate_tc()
        Gst = for_kin.GST(theta)

        pos = mathlib.get_position_from_matrix(Gst[len(Gst) - 1])
        quat = mathlib.Quaternion.from_matrix(Gst[len(Gst) - 1])
        rad = quat.to_euler()

        s = ""
        for p in theta:
            s += "%.8f " % p
        for p in pos:
            s += "%.8f " % p
        for p in rad:
            s += "%.8f " % p

        f.write(s+"\n")

    f.close()