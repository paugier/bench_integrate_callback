import numpy as np
from scipy.integrate import odeint

from transonic.util import timeit_verbose as timeit
import numba

from transonic import jit, wait_for_all_extensions


def lorenz(u, t):
    x, y, z = u
    return 10.0 * (y - x), x * (28.0 - z) - y, x * y - 2.66 * z


lorenz_pythran = jit(native=True, xsimd=True)(lorenz)
lorenz_numba = numba.jit(lorenz)

u0 = (1.0, 0.0, 0.0)

lorenz_pythran(u0, 0)
lorenz_numba(u0, 0)
wait_for_all_extensions()

# tspan = (0.0, 100.0)
t = np.linspace(0, 100, 1001)
sol = odeint(lorenz, u0, t)

norm = timeit("odeint(lorenz, u0, t, rtol = 1e-8, atol=1e-8)", globals=locals())
timeit(
    "odeint(lorenz_pythran, u0, t, rtol = 1e-8, atol=1e-8)",
    globals=locals(),
    norm=norm,
)
timeit(
    "odeint(lorenz_numba, u0, t, rtol = 1e-8, atol=1e-8)",
    globals=locals(),
    norm=norm,
)
