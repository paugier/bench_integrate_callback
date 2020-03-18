push!(LOAD_PATH, "./")

using Statistics
using stiffness

const S = StiffnessData()
const x = Float64[0., 1., 0.]
const y = Float64[0., 0., 1.]
const m = zeros(21)

nb_runs = 200

times = zeros(nb_runs)

for irun in 1:nb_runs
    times[irun] = @elapsed stiffness.op!(S, x, y, m)
end

println(median(times) * 1e6, " µs")
