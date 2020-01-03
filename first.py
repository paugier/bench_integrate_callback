import numpy as np
from scipy.integrate import odeint

from transonic.util import timeit_verbose as timeit
import numba

from transonic import jit


@jit(native=True, xsimd=True)
def f(u, t):
    print(u)
    x, y, z = u
    return 10.0 * (y - x), x * (28.0 - z) - y, x * y - 2.66 * z


u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 100.0)
t = np.linspace(0, 100, 1001)
sol = odeint(f, u0, t)
# def time_func():

#timeit("odeint(f, u0, t, rtol = 1e-8, atol=1e-8)", globals=locals())
