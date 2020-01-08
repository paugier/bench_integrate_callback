"""
pypy3 purepython_callback.py
1.55e-03 s

python purepython_callback.py
3.01e-02 s

- PyPy is 19 times faster than CPython.
- PyPy is 1.6 times faster than Julia.

"""

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


def call_function(func):
    u = (1.0, 0.0, 0.0)
    dt = 0.1
    times = [dt * it for it in range(int(1e5))]
    for time in times:
        u = func(time, u)


call_function(rober)

tiv("call_function(rober)", globals=locals(), total_duration=10)
