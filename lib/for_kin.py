from . import mathlib
import numpy as np
from .const import joint_axis, joint_translate


def translate_to_shoulder_frame(pos):
    M = mathlib.translate_matrix2(joint_translate[0])
    Rm = np.identity(4, dtype=np.float32)
    Rm[0:3, 0:3] = mathlib.eular_to_rotation_matrix2(joint_axis[1])
    M = np.dot(M, Rm)
    M = np.dot(M, mathlib.translate_matrix2(joint_translate[1]))

    pos = np.dot(np.linalg.inv(M), [pos[0], pos[1], pos[2], 1])

    return [pos[0], pos[1], pos[2]]


def GST(theta):
    j = theta

    joint_angles = np.array([
        [0,     0,      0   ], # mount->s0
        [0,     0,      0   ], # mount->s0
        [0,     0,      j[0]], # mount->s0
        [0,     0,      j[1]], # s0->s1
        [0,     0,      j[2]], # s1->e0
        [0,     0,      j[3]], # e0->e1
        [0,     0,      j[4]], # e1->w0
        [0,     0,      j[5]], # w0->w1
        [0,     0,      j[6]], # w1->w2
        [0,     0,      0   ], # w2->hand
        [0,     0,      0   ]  # hand->gripper
    ], dtype=np.float32)

    _joint_translate = joint_translate.copy()
    _joint_axis = joint_axis.copy()

    # Set Axis
    joint_angles += _joint_axis

    start = np.array([0, 0, 0])
    end = start
    pre = np.eye(3)
    ret = []
    for i in range(0, 10):
        rm = np.dot(pre, mathlib.eular_to_rotation_matrix(joint_angles[i][0], joint_angles[i][1], joint_angles[i][2]))
        end = start+np.dot(rm, _joint_translate[i])

        TF = np.identity(4)
        TF[0:3,0:3] = rm
        TF[0:3,3] = start
        ret.append(TF)

        start = end

        pre = rm


    TF = np.identity(4)
    TF[0:3,0:3] = pre
    TF[0:3,3] = start
    ret.append(TF)

    return ret
