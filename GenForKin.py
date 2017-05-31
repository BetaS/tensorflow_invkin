import numpy as np
from lib import for_kin, plotlib, mathlib
import time
import math
from lib.const import joint_name, joint_limit
import random

def random_joints():
    q = []
    for i in range(7):
        q.append(random.uniform(math.radians(joint_limit[i][0]), math.radians(joint_limit[i][1])))

    return q

def generate_tc():
    theta = random_joints()
    Gst = for_kin.GST(theta)

    pos = mathlib.get_position_from_matrix(Gst[len(Gst) - 1])
    quat = mathlib.Quaternion.from_matrix(Gst[len(Gst) - 1])
    rad = quat.to_euler()

    result = []
    for p in theta:
        result.append(math.degrees(p))
    for p in pos:
        result.append(p*1000)
    for p in rad:
        result.append(math.degrees(p))

    return result

if __name__ == "__main__":
    f = open("testset2.txt", "w")
    for i in range(10000):
        case = generate_tc()
        s = ""
        for p in case:
            s += "%.8f " % p

        f.write(s+"\n")

    f.close()
