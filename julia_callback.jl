#=

julia julia_callback.jl
  2.457 ms (100002 allocations: 10.68 MiB)

=#
using BenchmarkTools

function rober(t, u)
    k1 = 0.04
    k2 = 3e7
    k3 = 1e4
    y1, y2, y3 = u
    dy1 = -k1 * y1 + k3 * y2 * y3
    dy2 = k1 * y1 - k2 * y2 * y2 - k3 * y2 * y3
    dy3 = k2 * y2 * y2
    return [dy1, dy2, dy3]
end

rober(0, [1.0, 0.0, 0.0])

function call_function(func)
    u = [1.0, 0.0, 0.0]
    dt = 0.1
    times = dt * (0:1e5)
    for time in times
        u = func(time, u)
    end
end

@btime call_function(rober)