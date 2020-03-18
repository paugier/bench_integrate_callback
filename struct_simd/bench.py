import numpy as np

from transonic import boost

A = "float[]"


@boost
class Stiffness:

    __slots__ = ["grads", "grads3d", "gq", "gq3d"]

    grads: "float[:,:]"
    grads3d: "float[:,:,:]"
    gq: "float[:,:]"
    gq3d: "float[:,:,:]"

    def __init__(self):
        self.grads = np.empty((18, 2))
        self.grads3d = np.empty((6, 3, 2))

        self.gq = np.array(
            [
                [-1.0, -1.0],
                [1.0, 1.0],
                [-1.0, -1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, -1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, -2.0],
                [-2.0, -2.0],
                [2.0, 0.0],
                [0.0, 2.0],
                [2.0, 2.0],
                [2.0, 0.0],
                [0.0, 2.0],
                [-2.0, -2.0],
                [-2.0, 0.0],
            ]
        )

        self.gq3d = self.gq.flatten().reshape(self.grads3d.shape)

    @boost
    def compute(self, x: A, y: A, m: A):
        grads = self.grads
        gq = self.gq

        a11 = -y[0] + y[2]
        a12 = y[0] - y[1]
        a21 = x[0] - x[2]
        a22 = -x[0] + x[1]

        for i in range(18):
            grads[i, 0] = a11 * gq[i, 0] + a12 * gq[i, 1]
            grads[i, 1] = a21 * gq[i, 0] + a22 * gq[i, 1]
        # this seems slower:
        # grads[:, 0] = a11 * gq[:, 0] + a12 * gq[:, 1]
        # grads[:, 1] = a21 * gq[:, 0] + a22 * gq[:, 1]

        det = -(x[1] - x[2]) * y[0] + (x[0] - x[2]) * y[1] - (x[0] - x[1]) * y[2]
        dv = 1.0 / (6.0 * det)
        ii = 0
        for i in range(6):
            i3 = 3 * i
            for j in range(i + 1):
                j3 = 3 * j
                m[ii] = dv * (
                    grads[i3 : i3 + 3, 0] @ grads[j3 : j3 + 3, 0]
                    + grads[i3 : i3 + 3, 1] @ grads[j3 : j3 + 3, 1]
                )
                ii += 1

    @boost
    def compute_3d(self, x: A, y: A, m: A):

        grads = self.grads3d
        gq = self.gq3d

        a11 = -y[0] + y[2]
        a12 = y[0] - y[1]
        a21 = x[0] - x[2]
        a22 = -x[0] + x[1]

        for f in range(6):
            for p in range(3):
                grads[f, p, 0] = a11 * gq[f, p, 0] + a12 * gq[f, p, 1]
                grads[f, p, 1] = a21 * gq[f, p, 0] + a22 * gq[f, p, 1]

        det = -(x[1] - x[2]) * y[0] + (x[0] - x[2]) * y[1] - (x[0] - x[1]) * y[2]
        dv = 1.0 / (6.0 * det)
        ii = 0
        for i in range(6):
            for j in range(i + 1):
                s = 0.0
                for k in range(3):
                    s += (
                        grads[i, k, 0] * grads[j, k, 0]
                        + grads[i, k, 1] * grads[j, k, 1]
                    )
                m[ii] = dv * s
                ii += 1


if __name__ == "__main__":

    from transonic.util import timeit
    from functools import partial

    stiffness = Stiffness()
    x = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    m = np.zeros(21)

    stiffness.compute(x, y, m)
    m_3d = np.zeros(21)

    stiffness.compute_3d(x, y, m_3d)
    assert np.allclose(m, m_3d)

    timeit = partial(timeit, globals=locals(), total_duration=8)


    # time = timeit("stiffness.compute(x, y, m)", globals=locals())
    # print(f"{time * 1e6:.2f} µs")
    time = timeit(
        "stiffness.compute_3d(x, y, m)"
    )
    print(f"{time * 1e6:.2f} µs")

    try:
        from __pythran__.bench_e8c958181e03f6fe586ca36dbe63fe01 import __for_method__Stiffness__compute_3d
    except ImportError:
        from __pythran__.bench_16ca22d6c4f7271d975a6bac0b0cfb62 import __for_method__Stiffness__compute_3d
    s = stiffness

    time = timeit(
        "__for_method__Stiffness__compute_3d(s.gq3d, s.grads3d, x, y, m)"
    )
    print(f"{time * 1e6:.2f} µs")

    grads = stiffness.grads3d
    gq= stiffness.gq3d

    time = timeit(
        "__for_method__Stiffness__compute_3d(gq, grads, x, y, m)"
    )
    print(f"{time * 1e6:.2f} µs")
