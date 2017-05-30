import math
import random


if __name__ == "__main__":
    f = open("testset_arcsin.txt", "w")
    for i in range(1000):
        theta = random.uniform(-math.pi, math.pi)
        pos = [math.sin(theta), math.cos(theta)]

        f.write("%f %f %f\n"%(theta, pos[0], pos[1]))

    f.close()
