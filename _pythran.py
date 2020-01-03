import numpy as np

# pythran export capsule f(float, float)
def f(x, t):
    return 10.0 * x * np.cos(t)


# pythran export capsule integrand(float)
def integrand(x):
    return 10.0 * x * np.cos(x)
