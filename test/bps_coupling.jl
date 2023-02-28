include("../src/pdmps/BPScoupling.jl")
include("../src/generic_couplings/thorisson.jl")
using Random
using CoupledPDMPs
using Test
using Plots: plot, plot!

function ∇U(x::Vector)
    x
end

H = [1 0; 0  1]
sampler = BPS_coupling(∇U, H, .5)

Random.seed!(1)
x1 = randn(2); x2 = randn(2)
coupledstate = sampler.init(x1, x2)
sampler.onestep_couple(coupledstate)

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

function get_x1(cstate::BPScoupledstate)
    cstate.state1.skeleton.x
end
function get_x2(cstate::BPScoupledstate)
    cstate.state2.skeleton.x
end


samples = sample_coupledpdmp(sampler.onestep_couple, coupledstate, 500)


samples_x1 = hcat(get_x1.(samples)...)'
samples_x2 = hcat(get_x2.(samples)...)'

plot(samples_x1[:,1], samples_x1[:,2])
plot(samples_x2[:,1], samples_x2[:,2])
plot!(samples_x2[:,1], samples_x2[:,2])