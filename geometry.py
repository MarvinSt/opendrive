import numpy as np
from scipy.special import fresnel


def generate_triangle_face_indices(n: int):
    """Helper function to generate triangle face indices for a given length of vertices.
    The vertices are alternating along the length of the lane [inner, outer, inner, outer, ...]

    Args:
        n (int): _description_
    """

    fv = []
    i = 0
    while i + 3 < n:
        f = [i, i + 1, i + 2]
        fv.append(f)
        f = [i + 1, i + 3, i + 2]
        fv.append(f)
        i += 2

    return fv


def get_normal(v):
    """Return the normal vector to the given input vector, by swapping y/z

    Args:
        v (_type_): lateral vector
    """
    return np.array([v[0], v[2], v[1]])


def Rotate(x: np.float64, y: np.float64, z: np.float64, rx: np.float64, ry: np.float64, rz: np.float64):
    """
    Assumed order of rotation (TBC!): yaw, pitch, roll

    Args:
        x (np.float64): x-coordinate
        y (np.float64): y-coordinate
        z (np.float64): z-coordinate
        rx (np.float64): x-axis rotation angle (roll)
        ry (np.float64): y-axis rotation angle (pitch)
        rz (np.float64): z-axis rotation angle (yaw)

    Returns:
        np.array([x, y, z]): vector with the rotated coordinates
    """
    v = np.array([x, y, z])

    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(rx), -np.sin(rx)],
                   [0.0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0.0, np.sin(ry)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(ry), 0.0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0.0],
                   [np.sin(rz), np.cos(rz), 0.0],
                   [0.0, 0.0, 1.0]])

    return Rx.dot(Ry.dot(Rz.dot(v.copy())))


def RotateX(v, rx: np.float64):
    """
    Rotate vector around a single axis

    Args:
        x (np.float64): x-coordinate
        y (np.float64): y-coordinate
        z (np.float64): z-coordinate
        rx (np.float64): x-axis rotation angle (roll)

    Returns:
        np.array([x, y, z]): vector with the rotated coordinates
    """
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(rx), -np.sin(rx)],
                   [0.0, np.sin(rx), np.cos(rx)]])

    return Rx.dot(v.copy())


def RotateY(v, ry: np.float64):
    """
    Rotate vector around a single axis

    Args:
        x (np.float64): [x-coordinate, y-coordinate, z-coordinate]
        ry (np.float64): y-axis rotation angle (pitch)

    Returns:
        np.array([x, y, z]): vector with the rotated coordinates
    """
    Ry = np.array([[np.cos(ry), 0.0, np.sin(ry)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(ry), 0.0, np.cos(ry)]])

    return Ry.dot(v.copy())


def RotateZ(v, rz: np.float64):
    """
    Rotate vector around a single axis

    Args:
        x (np.float64): [x-coordinate, y-coordinate, z-coordinate]
        rz (np.float64): z-axis rotation angle (yaw)

    Returns:
        np.array([x, y, z]): vector with the rotated coordinates
    """
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0.0],
                   [np.sin(rz), np.cos(rz), 0.0],
                   [0.0, 0.0, 1.0]])

    return Rz.dot(v.copy())


class Poly3():
    a: np.float64
    b: np.float64
    c: np.float64
    d: np.float64

    s: np.float64
    t: np.float64

    def __init__(self, s, a, b, c, d, t=None):
        self.s = np.float64(s)
        self.a = np.float64(a)
        self.b = np.float64(b)
        self.c = np.float64(c)
        self.d = np.float64(d)
        self.t = np.float64(t)

    def eval_s(self, s):
        ds = s - self.s
        assert ds >= 0, "s is before start interval"
        return np.float64(self.a + (self.b + (self.c + self.d * ds) * ds) * ds)

    def eval_t(self, t):
        dt = t - self.t
        assert dt >= 0, "t is before start interval"
        return np.float64(self.a + (self.b + (self.c + self.d * dt) * dt) * dt)


class EulerSpiral(object):

    def __init__(self, gamma, kappa):
        self._gamma = gamma
        self._kappa = kappa

    @ staticmethod
    def create(length, curvStart, curvEnd):
        return EulerSpiral(1 * (curvEnd - curvStart) / length, curvStart)

    def get_tangent(self, s, x0=0, y0=0, theta0=0):
        # Tangent at each point
        return self._gamma * s**2 / 2.0 + self._kappa * s + theta0

    def get_xyt(self, s, x0=0, y0=0, theta0=0):

        # Start
        C0 = x0 + 1j * y0

        if self._gamma == 0 and self._kappa == 0:
            # Straight line
            Cs = C0 + np.exp(1j * theta0) * s

        elif self._gamma == 0 and self._kappa != 0:
            # Arc
            Cs = C0 + np.exp(1j * theta0) / self._kappa * \
                (np.sin(self._kappa * s) + 1j * (1 - np.cos(self._kappa * s)))

        else:
            # Fresnel integrals
            Sa, Ca = fresnel((self._kappa + self._gamma * s) /
                             np.sqrt(np.pi * np.abs(self._gamma)))
            Sb, Cb = fresnel(
                self._kappa / np.sqrt(np.pi * np.abs(self._gamma)))

            # Euler Spiral
            Cs1 = np.sqrt(np.pi / np.abs(self._gamma)) * \
                np.exp(1j * (theta0 - self._kappa**2 / 2 / self._gamma))
            Cs2 = np.sign(self._gamma) * (Ca - Cb) + 1j * Sa - 1j * Sb

            Cs = C0 + Cs1 * Cs2

        # Tangent at each point
        theta = self._gamma * s**2 / 2.0 + self._kappa * s + theta0

        return (Cs.real, Cs.imag, theta)


class Geometry():
    s: np.float64
    x: np.float64
    y: np.float64
    hdg: np.float64
    length: np.float64
    spiral: EulerSpiral

    def __init__(self, s, x, y, hdg, length, curvStart, curvEnd):
        self.s = np.float64(s)
        self.x = np.float64(x)
        self.y = np.float64(y)
        self.hdg = np.float64(hdg)
        self.length = np.float64(length)
        curvStart = np.float64(curvStart)
        curvEnd = np.float64(curvEnd)

        self.spiral = EulerSpiral.create(self.length, curvStart, curvEnd)

    def get_start_end(self):
        return (self.s, self.s+self.length)

    # def in_range(self, s):
    #     return not (s < self.s or s > self.s + self.length)

    def get_xyt(self, s):
        (x, y, t) = self.spiral.get_xyt(s - self.s, self.x, self.y, self.hdg)
        return (x, y, t)

    def get_grad(self, s):
        t = self.spiral.get_tangent(s - self.s, self.x, self.y, self.hdg)
        return (np.sin(t), np.cos(t))

    @staticmethod
    def Line(s, x, y, hdg, length):
        return Geometry(s, x, y, hdg, length, 0, 0)

    @staticmethod
    def Arc(s, x, y, hdg, length, curvature):
        return Geometry(s, x, y, hdg, length, curvature, curvature)

    @staticmethod
    def Spiral(s, x, y, hdg, length, curvStart, curvEnd):
        return Geometry(s, x, y, hdg, length, curvStart, curvEnd)


if __name__ == "__main__":
    """
    Test the rotation matrices. 
    """
    v = Rotate(1.0, 1.0, 1.0, np.deg2rad(90), np.deg2rad(0), np.deg2rad(0))
    print(v)
    v = Rotate(1.0, 1.0, 1.0, np.deg2rad(0), np.deg2rad(90), np.deg2rad(0))
    print(v)
    v = Rotate(1.0, 1.0, 1.0, np.deg2rad(0), np.deg2rad(0), np.deg2rad(90))
    print(v)
