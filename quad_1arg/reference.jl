
using QuadGK

using BenchmarkTools


function integrand(x)
    return exp(10.0 * x * cos(x))
end


@btime quadgk(integrand, 0, 10, rtol=1.49e-08, atol=1.49e-08)
