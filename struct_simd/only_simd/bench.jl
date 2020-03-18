using Statistics
using Printf



function compute_simd(x::Array{Float64, 1}, y::Array{Float64, 1},
                 grads::Array{Float64, 3}, out::Array{Float64, 1})

    det = -(x[2] - x[3])*y[1] + (x[1] - x[3])*y[2] - (x[1] - x[2])*y[3]
    dv = 1.0 / (6.0 * det)
    ii = 1
    for i=1:6
        for j=1:i
            s = 0.0
            @simd for k=1:3
                s += grads[1, k, i] * grads[1, k, j] + grads[2, k, i] * grads[2, k, j]
            end
            out[ii] = dv * s
            ii += 1
        end
    end

    return out
end

function compute(x::Array{Float64, 1}, y::Array{Float64, 1},
                 grads::Array{Float64, 3}, out::Array{Float64, 1})

    det = -(x[2] - x[3])*y[1] + (x[1] - x[3])*y[2] - (x[1] - x[2])*y[3]
    dv = 1.0 / (6.0 * det)
    ii = 1
    for i=1:6
        for j=1:i
            s = 0.0
            for k=1:3
                s += grads[1, k, i] * grads[1, k, j] + grads[2, k, i] * grads[2, k, j]
            end
            out[ii] = dv * s
            ii += 1
        end
    end

    return out
end


const x = Float64[0., 1., 0.]
const y = Float64[0., 0., 1.]

a11 = -y[1] + y[3]
a12 =  y[1] - y[2]
a21 =  x[1] - x[3]
a22 = -x[1] + x[2]

gq = reshape(
    [-1.0,-1.0,1.0,1.0,-1.0,-1.0,1.0,0.0,1.0,0.0,-1.0,0.0,0.0,
     -1.0,0.0,1.0,0.0,1.0,0.0,-2.0,-2.0,-2.0,2.0,0.0,0.0,2.0,2.0,
     2.0,2.0,0.0,0.0,2.0,-2.0,-2.0,-2.0,0], 2, 3, 6)

grads = zeros(2, 3, 6)

for f=1:6
    for p=1:3
        grads[1, p, f] = a11 * gq[1, p, f] + a12 * gq[2, p, f]
        grads[2, p, f] = a21 * gq[1, p, f] + a22 * gq[2, p, f]
    end
end


result = zeros(21)

nb_runs = 1000

times = zeros(nb_runs)

for irun in 1:nb_runs
    times[irun] = @elapsed compute(x, y, grads, result)
end

time_without_simd = median(times)

@printf("Julia without simd %.3f µs\n", time_without_simd * 1e6)

for irun in 1:nb_runs
    times[irun] = @elapsed compute_simd(x, y, grads, result)
end

time_with_simd = median(times)

@printf("Julia with simd    %.3f µs\n", time_with_simd * 1e6)


@printf("ratio (without simd)/(with simd): %.2f\n", time_without_simd / time_with_simd)