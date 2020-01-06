import numpy as np
from scipy.integrate import quad

from scipy import LowLevelCallable

import numba
from transonic.util import timeit_verbose as timeit
from transonic import jit, boost

from _pythran import integrand as integrand_capsule


def integrand(x):
    return 10.0 * x * np.cos(x)


integrand_transonic_jit = jit(native=True)(integrand)

# transonic bug (should work)!
# integrand_transonic_boost = boost(integrand)


@boost
def integrand_transonic_boost(x: float):
    return 10.0 * x * np.cos(x)


integrand_numba = numba.njit(integrand)
integrand_numba_cfunc = numba.cfunc("float64(float64)")(integrand)


ll_callable = LowLevelCallable(integrand_capsule, signature="double (double)")

methods = {
    "no acceleration": "integrand",
    "numba": "integrand_numba",
    "numba cfunc": "integrand_numba_cfunc.ctypes",
    "transonic_jit": "integrand_transonic_jit",
    "transonic_boost": "integrand_transonic_boost",
    "pythran capsule": "ll_callable",
}
norm = None
for name, key in methods.items():
    print(name)
    result = timeit(f"quad({key}, 0, 10)", globals=locals(), norm=norm)
    if norm is None:
        norm = result
