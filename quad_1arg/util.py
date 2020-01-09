import numpy as np

# pythran export capsule integrand(float)
def integrand(x):
    return np.exp(10.0 * x * np.cos(x))
