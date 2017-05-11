import math as math
import numpy as np

class Quaternion:
    def __init__(self, x, y, z, w):
        self._x = x
        self._y = y
        self._z = z
        self._w = w

    @classmethod
    def from_xyzw(cls, l):
        return Quaternion(l[0], l[1], l[2], l[3])

    @classmethod
    def from_wxyz(cls, l):
        return Quaternion(l[1], l[2], l[3], l[0])

    @classmethod
    def from_matrix(cls, m):
        tr = m[0][0] + m[1][1] + m[2][2]

        if tr > 0:
            S = math.sqrt(tr+1.0) * 2 # S=4*qw

            w = 0.25 * S
            x = (m[2][1] - m[1][2]) / S
            y = (m[0][2] - m[2][0]) / S
            z = (m[1][0] - m[0][1]) / S
        elif (m[0][0] > m[1][1]) and (m[0][0] > m[2][2]):
            S = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2 # S=4*qx
            w = (m[2][1] - m[1][2]) / S
            x = 0.25 * S
            y = (m[0][1] + m[1][0]) / S
            z = (m[0][2] + m[2][0]) / S
        elif m[1][1] > m[2][2]:
            S = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2 # S=4*qy
            w = (m[0][2] - m[2][0]) / S
            x = (m[0][1] + m[1][0]) / S
            y = 0.25 * S
            z = (m[1][2] + m[2][1]) / S
        else:
            S = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2 # S=4*qz
            w = (m[1][0] - m[0][1]) / S
            x = (m[0][2] + m[2][0]) / S
            y = (m[1][2] + m[2][1]) / S
            z = 0.25 * S

        return Quaternion(x,y,z,w)

    @classmethod
    def from_euler(cls, euler):
        pass

    def __str__(self):
        return "x = "+str(self._x)+", y = "+str(self._y)+", z = "+str(self._z)+", w = "+str(self._w)

    def __mul__(self, q):
        w1, x1, y1, z1 = (self._w, self._x, self._y, self._z)
        w2, x2, y2, z2 = (q._w, q._x, q._y, q._z)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

        return Quaternion(x, y, z, w)

    def inv(self):
        S = self._x**2+self._y**2+self._z**2+self._w**2

        return Quaternion(-self._x/S, -self._y/S, -self._z/S, self._w/S)

    def to_euler(self):
        rx = math.atan2((2*(self._x*self._w+self._y*self._z)), 1-2*(self._x**2+self._y**2))
        ry = math.asin(2*(self._w*self._y-self._x*self._z))
        rz = math.atan2((2*(self._x*self._y+self._z*self._w)), 1-2*(self._y**2+self._z**2))

        return [rx, ry, rz]

    def to_euler_degree(self):
        r = self.to_euler()

        return [math.degrees(r[0]), math.degrees(r[1]), math.degrees(r[2])]

    def to_rotation_matrix(self, size=3):
        ret = np.identity(size, dtype=np.float32)

        xx = self._x * self._x
        xy = self._x * self._y
        xz = self._x * self._z
        xw = self._x * self._w

        yy = self._y * self._y
        yz = self._y * self._z
        yw = self._y * self._w

        zz = self._z * self._z
        zw = self._z * self._w

        m00 = 1 - 2 * ( yy + zz )
        m01 =     2 * ( xy - zw )
        m02 =     2 * ( xz + yw )

        m10 =     2 * ( xy + zw )
        m11 = 1 - 2 * ( xx + zz )
        m12 =     2 * ( yz - xw )

        m20 =     2 * ( xz - yw )
        m21 =     2 * ( yz + xw )
        m22 = 1 - 2 * ( xx + yy )

        ret[:3, :3] = np.array([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]], dtype=np.float32)

        return ret

    def distance(self, dist_x, dist_y, dist_z):
        rot = self.to_rotation_matrix()
        #rot = np.transpose(rot)
        return np.dot(rot, np.array([dist_x, dist_y, dist_z], dtype=np.float32))

    def decompose(self):
        return [self._x, self._y, self._z, self._w]


def get_position_from_matrix(gst):
    return [gst[0][3], gst[1][3], gst[2][3]]

def tr2diff(t1, t2):
    IR = np.dot(np.linalg.inv(t2), t1)
    DT = t1[0:3,3]-t2[0:3,3]
    DR = Quaternion.from_matrix(IR[0:3,0:3])
    DR = DR.to_euler()

    return np.array([DT[0], DT[1], DT[2], DR[0], DR[1], DR[2]], dtype=np.float32)

# Get sum of square from data list
def sum_of_square(data):
    return sum(map(lambda x: x**2, data))

# Get variance from data list
def var(data):
    def _var(sum_of_square, sum, len):
        return (sum_of_square - sum**2 / len) / len

    return _var(sum_of_square(data), sum(data), len(data))*10

def length_vector(vec):
    s = sum(vec)
    return s/len(vec)

def mag_vector(vec):
    s = sum_of_square(vec)
    return math.sqrt(s)

def unit_vector(vec):
    mag = mag_vector(vec)
    return [vec[0]/mag, vec[1]/mag, vec[2]/mag]

def translate_matrix(x, y, z):
    return np.array([
        [1,         0,      0,      x],
        [0,         1,      0,      y],
        [0,         0,      1,      z],
        [0,         0,      0,      1]], dtype=np.float32)

def translate_matrix2(l):
    return translate_matrix(l[0], l[1], l[2])

def eular_to_rotation_matrix(R, P, Y):
    RX = np.array([
        [1,          0,             0],
        [0, math.cos(R), -math.sin(R)],
        [0, math.sin(R),  math.cos(R)]], dtype=np.float32)

    RY = np.array([
        [math.cos(P),  0,  math.sin(P)],
        [          0,  1,            0],
        [-math.sin(P), 0,  math.cos(P)]], dtype=np.float32)

    RZ = np.array([
        [math.cos(Y), -math.sin(Y), 0],
        [math.sin(Y),  math.cos(Y), 0],
        [0,            0,           1]], dtype=np.float32)

    return np.dot(np.dot(RX,RY),RZ)

def eular_to_rotation_matrix2(l):
    return eular_to_rotation_matrix(l[0], l[1], l[2])

def eular_to_dir_vector(R, P, Y):
    x = math.cos(Y)*math.cos(P)
    y = math.sin(Y)*math.cos(P)
    z = math.sin(P)

    return [x,y,z]

def rotation_matrix_from_axis(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2)
    b, c, d = -axis*math.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac), 0],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab), 0],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc, 0],
                     [0, 0, 0, 1]])

def line_intersect_skewed(l1_start, l1_end, l2_start, l2_end):
    s1 = l1_start
    s2 = l2_start
    v1 = unit_vector(l1_end-l1_start)
    v2 = unit_vector(l2_end-l2_start)

    r = np.array([0,0,0], dtype=np.float32)
    t = np.array([0,0,0], dtype=np.float32)

    m = np.identity(3, dtype=np.float32)

    # Precomputed values
    v1dotv2 = np.dot(v1, v2)
    v1p2 = np.dot(v1, v1)
    v2p2 = np.dot(v2, v2)

    # Solving matrix
    m[0][0] = -v2p2
    m[1][0] = -v1dotv2
    m[0][1] = v1dotv2
    m[1][1] = v1p2

    # Projected vector
    r[0] = np.dot((s2 - s1), v1)
    r[1] = np.dot((s2 - s1), v2)

    # precomputed value
    d = 1.0 / (v1dotv2 * v1dotv2 - v1p2 * v2p2)

    # Compute time values
    t = d * np.dot(m, r)

    # Compute intersected points on each lines
    p1 = s1 + np.dot(t[0], v1)
    p2 = s2 + np.dot(t[1], v2)

    return (p1+p2)/2, dist(p1, p2)

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def dist(a, b):
    if len(a) != len(b):
       raise Exception('A and B must have same size')

    c = 0
    for i in range(len(a)):
        c += (a[i] - b[i])**2

    return math.sqrt(c)

def center_point(l):
    return center_point2(l[0], l[1], l[2], l[3])

def center_point2(a, b, c, d):
    cx = (a[0]+b[0]+c[0]+d[0])/4
    cy = (a[1]+b[1]+c[1]+d[1])/4
    cz = (a[2]+b[2]+c[2]+d[2])/4

    return [cx, cy, cz]

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect