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


function get_x1(state::Tuple)
    state[1].skeleton.x
end
function get_x2(state::Tuple)
    state[2].skeleton.x
end
function get_t1(state::Tuple)
    state[1].skeleton.t
end
function get_t2(state::Tuple)
    state[2].skeleton.t
end
using Plots: plot, plot!, scatter, scatter!

using Distributions
using Random
using Statistics
using StatsBase
using CoupledPDMPs
using Test
using Distributed
using SharedArrays

H = diagm([1.])

function ∇U(x::Vector)
    x
end

function h(pdmp_state::Skeleton)
    return pdmp_state.x[1]
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
function sample_event(kernel, coupledstate, N)
    states = Vector{typeof(coupledstate)}(undef, N)
    states[1] = coupledstate
    for i in 1:(N-1)
        println(i)
        states[i+1] = kernel(states[i]...)
    end
    return states
end

include("../src/pdmps/BPS_coupling.jl")
Random.seed!(1)
sampler = BPS_coupling(∇U, H, h, .1, 1.)
x1 = randn(1); x2 = randn(1) .+ 10
coupled_state = sampler.init(x1, x2)


# Run sampler event_kernel
Random.seed!(1)
event_state = (coupled_state.state_1.event_vec[1], coupled_state.state_2.event_vec[1])
samples2 = sample_event(sampler.onestep_event, event_state, 100)

samples_x1 = hcat(get_x1.(samples2)...)'
samples_x2 = hcat(get_x2.(samples2)...)'

plot(get_t1.(samples2), samples_x1[:,1])
plot!(get_t2.(samples2), samples_x2[:,1])
xlims!(0, 75)

# Run sampler discrete_kernel
Random.seed!(1)
samples = sample_coupledpdmp(sampler.onestep, coupled_state, 75*10)

samples_x1 = hcat(get_x1.(samples)...)'
samples_x2 = hcat(get_x2.(samples)...)'
scatter(get_t1.(samples), samples_x1[:,1],  markersize=2)
scatter!(get_t2.(samples), samples_x2[:,1],  markersize=2)
xlims!(0, 75)



## Estimator



K = 250
M = 100_000

Random.seed!(1)
τ, h_out, i_out = rhee_glynn(sampler.onestep, coupled_state, K, M, true)




