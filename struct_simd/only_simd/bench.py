import numpy as np

from transonic import boost

A1 = "float[]"


@boost
def compute(x: A1, y: A1, grads: "float[:, :, :]", out: A1):
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
            out[ii] = dv * s
            ii += 1


if __name__ == "__main__":

    from functools import partial
    from transonic.util import timeit

    x = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0])

    grads = np.empty((6, 3, 2))

    # fmt: off
    gq = np.array(
        [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 0.0, 1.0, 0.0, -1.0,
         0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -2.0, -2.0, -2.0,
         2.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 2.0, -2.0, -2.0,
         -2.0, 0.0]
    )
    # fmt: on

    gq = gq.reshape(grads.shape)
    a11 = -y[0] + y[2]
    a12 = y[0] - y[1]
    a21 = x[0] - x[2]
    a22 = -x[0] + x[1]

    for f in range(6):
        for p in range(3):
            grads[f, p, 0] = a11 * gq[f, p, 0] + a12 * gq[f, p, 1]
            grads[f, p, 1] = a21 * gq[f, p, 0] + a22 * gq[f, p, 1]

    result = np.zeros(21)

    compute(x, y, grads, result)

    timeit = partial(timeit, globals=locals(), total_duration=8)
    time = timeit("compute(x, y, grads, result)") * 1e6
    print(f"Pythran: {time:.2f} µs")

    from pathlib import Path
    import sys

    path_tmp_julia = Path("tmp_result_julia.txt")
    if not path_tmp_julia.exists():
        sys.exit()

    with open(path_tmp_julia) as file:
        txt = file.read()

    time_julia = float(txt.split("\n")[1].split(" ")[-2])

    print(f"Julia:   {time_julia:.2f} µs\nratio Pythran/Julia: {time/time_julia:.2f}")
