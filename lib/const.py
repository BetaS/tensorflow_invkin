import numpy as np

joint_num = 7

JOINT_BASE  = 0
JOINT_MOUNT = 1
JOINT_S0    = 2
JOINT_S1    = 3
JOINT_E0    = 4
JOINT_E1    = 5
JOINT_W0    = 6
JOINT_W1    = 7
JOINT_W2    = 8
JOINT_HAND  = 9

joint_translate = np.array([
    [0.025,     0.219,      0.108   ], # b->mount
    [0.056,     0,          0.011   ], # mount->s0
    [0.069,     0,          0.27    ], # s0->s1
    [0.102,     0,          0       ], # s1->e0
    [0.069,     0,          0.26242 ], # e0->e1
    [0.104,     0,          0       ], # e1->w0
    [0.01,      0,          0.271   ], # w0->w1
    [0.11597,   0,          0       ], # w1->w2
    [0,         0,          0.11355 ], # w2->hand
    [0,         0,          0.045   ]  # hand->gripper
], dtype=np.float32)

joint_axis = np.array([
    [0,      0,         0           ],
    [-0.002, 0.001,     0.780       ], # b->mount
    [0,      0,         0           ], # mount->s0
    [-1.571, 0,         0           ], # s0->s1
    [1.571,  1.571,     0           ], # s1->e0
    [-1.571, 0,         -1.571      ], # e0->e1
    [1.571,  1.571,     0           ], # e1->w0
    [-1.571, 0,         -1.571      ], # w0->w1
    [1.571,  1.571,     0           ], # w1->w2
    [0,      0,         0           ], # w2->hand
    [0,      0,         0           ]  # hand->gripper
], dtype=np.float32)