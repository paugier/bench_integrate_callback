import numpy as np
from scipy.integrate import quad

from scipy import LowLevelCallable

import numba
from transonic.util import timeit_verbose as timeit
from transonic import jit, boost

from _pythran import integrand as integrand_capsule


def integrand(x):
    return 10.0 * x * np.cos(x)

@boost
def integrand_transonic_boost(x):
    return 10.0 * x * np.cos(x)


integrand_transonic_jit = jit(native=True)(integrand)
# integrand_transonic_boost = boost(native=True)(integrand)
integrand_numba = numba.njit(integrand)

callable = LowLevelCallable(integrand_capsule, signature="double (double)")

methods = {
    "no acceleration": "integrand",
    "numba": "integrand_numba",
    "transonic_jit": "integrand_transonic_jit",
    "transonic_boost": "integrand_transonic_boost",
    "pythran capsule": "callable",
}
norm = None
for name, key in methods.items():
    print(name)
    result = timeit(f"quad({key}, 0, 10)", globals=locals(), norm=norm)
    if norm is None:
        norm = result
