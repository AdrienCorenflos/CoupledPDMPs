include("../src/pdmps/BPS_Efficient.jl")
using Random
using Test
using Plots: plot, plot!

function ∇U(x::Vector)
    x
end

H = [1 0; 0  1]

Random.seed!(123456)
sampler = BPS(∇U, H, .5)
x_current = randn(2)
state = sampler.init(x_current)
sampler.onestep(state)

# Run the sampler (memory inefficient version)
function sample_pdmp(kernel, state, N)
    states = Vector{typeof(state)}(undef, N)
    states[1] = state
    for i in 1:(N-1)
        states[i+1] = kernel(states[i])
    end
    return states
end

function get_x(state::BPSstate)
    state.skeleton.x
end

Random.seed!(123456)
samples = sample_pdmp(sampler.onestep, state, 10_000)
samples_x = hcat(get_x.(samples)...)'

plot(samples_x[:,1], samples_x[:,2])
