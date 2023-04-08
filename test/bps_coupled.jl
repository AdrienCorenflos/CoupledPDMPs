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

function ∇U(x::Vector)
    x
end

function h(pdmp_state::Skeleton)
    return pdmp_state.x[1]
end

# Run the sampler
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
        states[i+1] = kernel(states[i]...)
    end
    return states
end

d = 50
diag_d = fill(1., d)
H = diagm(diag_d)

Random.seed!(1)
Δt = 10.
sampler = BPS_coupling(∇U, H, h, Δt, 1.)
x1 = randn(dim(H)) .* 10; x2 = randn(dim(H)) .* 10
v1 = randn(dim(H))
v2 = copy(v1)
coupled_state = sampler.init(x1, x2)

# Run sampler discrete_kernel
Random.seed!(1)
samples = sample_coupledpdmp(sampler.onestep, coupled_state, Int(ceil(3000/Δt)))

samples_x1 = hcat(get_x1.(samples)...)'
samples_x2 = hcat(get_x2.(samples)...)'
scatter(get_t1.(samples), samples_x1[:,1],  markersize=1,markerstrokewidth=0)
scatter!(get_t2.(samples), samples_x2[:,1],  markersize=1,markerstrokewidth=0)



## Estimator
K = 10
M = 1_000

Random.seed!(1)
τ, h_out, i_out = rhee_glynn(sampler.onestep, coupled_state, K, M, true)




