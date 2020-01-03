import numpy as np
from scipy.integrate import odeint

from transonic.util import timeit_verbose as timeit

# import numba

from transonic import jit

# @jit(native=True, xsimd=True)
def f(x, t):
    return 10.0 * x * np.cos(t)


x0 = 1.0
t = np.linspace(0, 4 * np.pi, 1001)
# sol = odeint(f, x0, t)
# def time_func():

timeit("odeint(f, x0, t, rtol = 1e-8, atol=1e-8)", globals=locals())
