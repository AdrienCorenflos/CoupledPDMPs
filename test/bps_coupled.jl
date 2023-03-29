include("../src/coupled_estimator.jl")
include("../src/generic_couplings/thorisson.jl")
include("../src/pdmps/BPS_coupling.jl")

function get_x1(cstate::BPScoupledstate)
    cstate.state_1.current.x
end
function get_x2(cstate::BPScoupledstate)
    cstate.state_2.current.x
end
function get_t1(cstate::BPScoupledstate)
    cstate.state_1.current.t
end
function get_t2(cstate::BPScoupledstate)
    cstate.state_2.current.t
end
using Plots: plot, plot!

using Distributions
using Random
using Statistics
using StatsBase
using CoupledPDMPs
using Test
using Distributed
using SharedArrays

H = [1 0; 0  1]

function ∇U(x::Vector)
    x
end

function h(pdmp_state::Skeleton)
    return pdmp_state.x
end

# Run the sampler (memory inefficient version)
function sample_coupledpdmp(kernel, coupledstate, N)
    states = Vector{typeof(coupledstate)}(undef, N)
    states[1] = coupledstate
    for i in 1:(N-1)
        println(i)
        states[i+1] = kernel(states[i])
    end
    return states
end

include("../src/pdmps/BPS_coupling.jl")
Random.seed!(2)
sampler = BPS_coupling(∇U, H, h, 0.1, 1.)
x1 = randn(2); x2 = randn(2)
state = sampler.init(x1, x2)
state1 = sampler.onestep(state)

samples = sample_coupledpdmp(sampler.onestep, state, 30)

samples_x1 = hcat(get_x1.(samples)...)'
samples_x2 = hcat(get_x2.(samples)...)'

plot(get_t1.(samples), samples_x1[:,1])
plot!(get_t2.(samples), samples_x2[:,1])

plot(samples_x1[:,1], samples_x1[:,2])
plot!(samples_x2[:,1], samples_x2[:,2])


Random.seed!(2)
sampler2 = BPS_coupling2(∇U, H, h, 0.1, 1.)
x1 = randn(2); x2 = randn(2)
astate = sampler2.init(x1, x2)

samples2 = sample_coupledpdmp(sampler2.onestep, astate, 50)

function get_x1(cstate::BPSalt)
    cstate.state_1.skeleton.x
end
function get_x2(cstate::BPSalt)
    cstate.state_2.skeleton.x
end
function get_t1(cstate::BPSalt)
    cstate.state_1.skeleton.t
end
function get_t2(cstate::BPSalt)
    cstate.state_2.skeleton.t
end
samples_x1 = hcat(get_x1.(samples2)...)'
samples_x2 = hcat(get_x2.(samples2)...)'

plot(get_t1.(samples2), samples_x1[:,1])
plot!(get_t2.(samples2), samples_x2[:,1])


plot(samples_x1[:,1], samples_x1[:,2])
plot!(samples_x2[:,1], samples_x2[:,2])
