"""
See end of https://github.com/JuliaDiffEq/SciPyDiffEq.jl/commit/8e623731387927fbe586291207bbb3fb7732f525

Actually no failure...

"""

from scipy.integrate import solve_ivp

from transonic.util import timeit_verbose as tiv


def rober(t, u):
    k1 = 0.04
    k2 = 3e7
    k3 = 1e4
    y1, y2, y3 = u
    dy1 = -k1 * y1 + k3 * y2 * y3
    dy2 = k1 * y1 - k2 * y2 * y2 - k3 * y2 * y3
    dy3 = k2 * y2 * y2
    return dy1, dy2, dy3


u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 1e5)
tiv("solve_ivp(rober, tspan, u0, t_eval=tspan)", globals=locals())
