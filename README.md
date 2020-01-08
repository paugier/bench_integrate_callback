# Benchmark Python callbacks small functions for integration

Integration and ODEs (scipy.integrate)

https://docs.scipy.org/doc/scipy/reference/integrate.html

https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html#faster-integration-using-low-level-callback-functions

https://github.com/JuliaDiffEq/SciPyDiffEq.jl

## Silly "callback tiny function"

### Julia comparison

julia julia_callback.jl
2.457 ms

### Without Numpy

pypy3 purepython_callback.py
1.55 ms

python purepython_callback.py
30.1 ms

- PyPy is 19 times faster than CPython.
- PyPy is 1.6 times faster than Julia.

### With Numpy

pypy3 numpy_callback.py
65.9 ms

python numpy_callback.py
32.3 ms

PyPy and CPython are super slow...

## A possible improvement: HPy

- https://morepypy.blogspot.com/2019/12/hpy-kick-off-sprint-report.html

- https://github.com/pyhandle/hpy