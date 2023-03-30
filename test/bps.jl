include("../src/coupled_estimator.jl")
include("../src/generic_couplings/thorisson.jl")
include("../src/pdmps/BPS_coupling.jl")

function get_x1(cstate::BPSstate)
    cstate.skeleton.x
end
function get_t1(cstate::BPSstate)
    cstate.skeleton.t
end
function get_x1(cstate::BPSDiscreteState)
    cstate.current.x
end
function get_t1(cstate::BPSDiscreteState)
    cstate.current.t
end
using Plots: plot, plot!, scatter

using Distributions
using Random
using Statistics
using StatsBase
using CoupledPDMPs
using Test
using Distributed
using SharedArrays

H = diagm([1.])#[1 0; 0  1]

function ∇U(x::Vector)
    x #.- 1.
end

function h(pdmp_state::Skeleton)
    return pdmp_state.x
end

function sample(kernel, state, N)
    states = Vector{typeof(state)}(undef, N)
    states[1] = state
    for i in 1:(N-1)
        println(i)
        states[i+1] = kernel(states[i])
    end
    return states
end

include("../src/pdmps/BPS.jl")
Random.seed!(1)
sampler = BPS(∇U, H, h, 1., 1.)
x1 = randn(1);# x2 = randn(2)
state = sampler.init(x1)
state1 = sampler.onestep(state)
state1e = sampler.onestep_event(state.event_vec[end])

Random.seed!(1)
samples = sample(sampler.onestep, state, 100)

Random.seed!(1)
samples_event = sample(sampler.onestep_event, state.event_vec[end], 100)

samples_x1 = hcat(get_x1.(samples)...)'
scatter(get_t1.(samples), samples_x1[:,1], markersize=1)

events_x = hcat(get_x1.(samples_event)...)'
plot!(get_t1.(samples_event), events_x[:,1])

plot(samples_x1[:,2], samples_x1[:,1])
plot!(events_x[:,2], events_x[:,1])




