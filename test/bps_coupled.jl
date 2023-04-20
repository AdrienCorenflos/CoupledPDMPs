include("../src/coupled_estimator.jl")
include("../src/pdmps/BPS_coupling.jl")
using Plots: plot, plot!, scatter, scatter!, boxplot, boxplot!, histogram

using Distributions
using Random
using StatsBase

function ∇U(x::Vector)
    x .- 1.5
end

d = 2
diag_d = fill(1., d)
H = diagm(diag_d)

Random.seed!(1)
Δt = 1.
sampler = BPS_coupling(∇U, H, Δt, 1.)
x1 = randn(dim(H)) .* 10; x2 = randn(dim(H)) .* 10
coupled_state = sampler.init(x1, x2)

# When time coupled
function couple_time(kernel, coupledstate)
    i = 1
    while !coupledstate[end]
        coupledstate = kernel(coupledstate...)
    end
    return coupledstate[1][1]
end

N = 500
res = zeros(Float64, N)
for i = 207:N
    Random.seed!(i)
    x1 = randn(dim(H)) .* 10; x2 = randn(dim(H)) .* 10
    coupled_state = sampler.init(x1, x2)
    println(i)
    res[i] = couple_time(sampler.onestep, coupled_state)
end

boxplot(res)



# Run the sampler
function sample_event(kernel, coupledstate, N)
    states = Vector{typeof(coupledstate)}(undef, N)
    states[1] = coupledstate
    for i in 1:(N-1)
        states[i+1] = kernel(states[i]...)
        #println(i)
    end
    return states
end

# Run sampler 
Random.seed!(207)
x1 = randn(dim(H)) .* 10; x2 = randn(dim(H)) .* 10
coupled_state = sampler.init(x1, x2)
samples = sample_event(sampler.onestep, coupled_state, 53976); ## Error on eval 53975 (ref time is 0. for PDMP 2)
samples[end][1]
samples[end][2]
samples[end][3]
function get_x1(coupled_state::Tuple)
    coupled_state[1][2]
end
function get_t1(coupled_state::Tuple)
    coupled_state[1][1]
end
function get_x2(coupled_state)
    coupled_state[2][2]
end
function get_t2(coupled_state::Tuple)
    coupled_state[2][1]
end

samples_x1 = hcat(get_x1.(samples)...)'
samples_x2 = hcat(get_x2.(samples)...)'

scatter(get_t1.(samples), samples_x1[:,1],  markersize=1,markerstrokewidth=0)
scatter!(get_t2.(samples), samples_x2[:,1],  markersize=1,markerstrokewidth=0)

inds = 52972:1:53972
plot(get_t1.(samples)[inds], samples_x1[inds,1])
plot!(get_t2.(samples)[inds], samples_x2[inds,1])

