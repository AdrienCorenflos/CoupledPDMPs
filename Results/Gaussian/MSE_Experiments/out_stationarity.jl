## Include the packages and samplers
using Distributions
using Random
using LinearAlgebra
using StatsBase
using Base.Threads
using JLD2
using CSV
using DataFrames
include("../../../src/pdmps/PDMP_discrete.jl")
include("../../../src/coupled_estimator_verb.jl")

function get_Gaussian(p::Int)
    function U(x::Vector)
        return sum(x .* x /2)
    end
    # Grad 
    function ∇U(x::Vector)
        return x
    end
    H = diagm(ones(p))
    Σ_inv = H;    Σ = H;    μ = zeros(p)
    return U, ∇U, H, Σ, Σ_inv, μ 
end

function get_estimate!(est1_out, est2_out, comp_out, discrete_sampler, K, M)
    x1 = 10.0 .* randn(p); x2 = 10.0 .* randn(p)
    coupled_state = discrete_sampler.init(x1, x2)
    
    (kern, (est1, est2), _, cs) = rhee_glynn(discrete_sampler.kernel, coupled_state, K, M)

    est1_out .= est1
    est2_out .= est2

    comp_out .= [kern, cs.num_grad_1, cs.num_grad_2, cs.num_grad_coupled]
    return Nothing
end

function non_coupled_estimator!(est1_out,est2_out, burn10_out, burn20_out,burn50_out, comp_stored, comp_budget, discrete_sampler)
    x1 = 10.0 .* randn(p); x2 = 10.0 .* randn(p)
    coupled_state = discrete_sampler.init(x1, x2)
    
    h = coupled_state.state_1.h
    h_burn10 = zero(h[1])
    h_burn20 = zero(h[1])
    h_burn50 = zero(h[1])

    burn10_start = Int(ceil(0.1 * comp_budget))
    burn20_start = Int(ceil(0.2 * comp_budget))
    burn50_start = Int(ceil(0.5 * comp_budget))


    for k in 1:comp_budget
        coupled_state = discrete_sampler.kernel(coupled_state)
        h = h .+ coupled_state.state_1.h
        if k > burn10_start
            h_burn10 .= h_burn10 .+ coupled_state.state_1.h[1]
        end
        if k > burn20_start
            h_burn20 .= h_burn20 .+ coupled_state.state_1.h[1]
        end
        if k > burn50_start
            h_burn50 .= h_burn50 .+ coupled_state.state_1.h[1]
        end
    end
    h = h ./ comp_budget
    h_burn10 ./= (comp_budget - burn10_start)
    h_burn20 ./= (comp_budget - burn20_start)
    h_burn50 ./= (comp_budget - burn50_start)

    cs = coupled_state.coupled_status
    comp_stored .= cs.num_grad_1+cs.num_grad_coupled

    est1_out .= h[1]
    est2_out .= h[2]

    burn10_out .= h_burn10
    burn20_out .= h_burn20
    burn50_out .= h_burn50
    return Nothing
end

# Functions of interest
function mn(x)
    x, x.^2
end

# Target
p = 20
U, ∇U, H, Σ, Σ_inv, μ = get_Gaussian(p)
Σ_sqrt = sqrt(Σ)

# Sampler
Δt = 15.0; λ = 1.0; ΔM = 5
sampler = BPS_coupling(∇U, H, Δt, ΔM, λ, mn)
discrete_sampler = DiscretePDMPCoupling(sampler)

## Experiment
sim_runs = 1:1000
K = 23
M = 10*K
comp = Array{Float64}(undef, length(sim_runs), 4) # kernel, grad
h_1 = Array{Float64}(undef, length(sim_runs), p)
h_2 = Array{Float64}(undef, length(sim_runs), p)

Random.seed!(1)
for l in sim_runs
    get_estimate!(view(h_1, l, :), view(h_2, l, :),
            view(comp, l, :), discrete_sampler, K, M)
end
print(quantile(comp[:,1,1],0.9)) # For 100 sims with seed 0  quantile = 22.29
num_kern = 2 * (comp[:,1,1] .- 1) + max.(1,M + 1 .-comp[:,1,1])

comp_single = Array{Float64}(undef, length(sim_runs), 4) # kernel, grad
h_1_single = Array{Float64}(undef, length(sim_runs), p)
h_2_single = Array{Float64}(undef, length(sim_runs), p)

h_burn10_single = Array{Float64}(undef, length(sim_runs), p)
h_burn20_single = Array{Float64}(undef, length(sim_runs), p)
h_burn50_single = Array{Float64}(undef, length(sim_runs), p)

Random.seed!(1)
for l in sim_runs
    non_coupled_estimator!(view(h_1_single, l, :), view(h_2_single, l, :), view(h_burn10_single, l, :),view(h_burn20_single, l, :),view(h_burn50_single, l, :),
            view(comp_single, l, :), num_kern[l], discrete_sampler)
end

### Check overall efficiency and start making table
eff_BPS_M10 = [sum(mean(h_1.^2, dims = 1)) / sum(mean(h_1_single.^2, dims = 1)),
            sum(mean(h_1.^2, dims = 1)) / sum(mean(h_burn10_single.^2, dims = 1)),
            sum(mean(h_1.^2, dims = 1)) / sum(mean(h_burn20_single.^2, dims = 1)),
            sum(mean(h_1.^2, dims = 1)) / sum(mean(h_burn50_single.^2, dims = 1))]

## Experiment
M = 20*K
comp = Array{Float64}(undef, length(sim_runs), 4) # kernel, grad
h_1 = Array{Float64}(undef, length(sim_runs), p)
h_2 = Array{Float64}(undef, length(sim_runs), p)

Random.seed!(1)
for l in sim_runs
    get_estimate!(view(h_1, l, :), view(h_2, l, :),
            view(comp, l, :), discrete_sampler, K, M)
end
num_kern = 2 * (comp[:,1,1] .- 1) + max.(1,M + 1 .-comp[:,1,1])
comp_single = Array{Float64}(undef, length(sim_runs), 4) 
h_1_single = Array{Float64}(undef, length(sim_runs), p)
h_2_single = Array{Float64}(undef, length(sim_runs), p)

h_burn10_single = Array{Float64}(undef, length(sim_runs), p)
h_burn20_single = Array{Float64}(undef, length(sim_runs), p)
h_burn50_single = Array{Float64}(undef, length(sim_runs), p)

Random.seed!(1)
for l in sim_runs
    non_coupled_estimator!(view(h_1_single, l, :), view(h_2_single, l, :), view(h_burn10_single, l, :),view(h_burn20_single, l, :),view(h_burn50_single, l, :),
            view(comp_single, l, :), num_kern[l], discrete_sampler)
end

### Check overall efficiency and start making table
eff_BPS_M20 = [sum(mean(h_1.^2, dims = 1)) / sum(mean(h_1_single.^2, dims = 1)),
            sum(mean(h_1.^2, dims = 1)) / sum(mean(h_burn10_single.^2, dims = 1)),
            sum(mean(h_1.^2, dims = 1)) / sum(mean(h_burn20_single.^2, dims = 1)),
            sum(mean(h_1.^2, dims = 1)) / sum(mean(h_burn50_single.^2, dims = 1))]
print("eff_BPS_M20")
print(eff_BPS_M20)

## Experiment
M = 50*K
comp = Array{Float64}(undef, length(sim_runs), 4) # kernel, grad
h_1 = Array{Float64}(undef, length(sim_runs), p)
h_2 = Array{Float64}(undef, length(sim_runs), p)

Random.seed!(1)
for l in sim_runs
    get_estimate!(view(h_1, l, :), view(h_2, l, :),
            view(comp, l, :), discrete_sampler, K, M)
end
num_kern = 2 * (comp[:,1,1] .- 1) + max.(1,M + 1 .-comp[:,1,1])
comp_single = Array{Float64}(undef, length(sim_runs), 4) 
h_1_single = Array{Float64}(undef, length(sim_runs), p)
h_2_single = Array{Float64}(undef, length(sim_runs), p)

h_burn10_single = Array{Float64}(undef, length(sim_runs), p)
h_burn20_single = Array{Float64}(undef, length(sim_runs), p)
h_burn50_single = Array{Float64}(undef, length(sim_runs), p)

Random.seed!(1)
for l in sim_runs
    non_coupled_estimator!(view(h_1_single, l, :), view(h_2_single, l, :), view(h_burn10_single, l, :),view(h_burn20_single, l, :),view(h_burn50_single, l, :),
            view(comp_single, l, :), num_kern[l], discrete_sampler)
end

### Check overall efficiency and start making table
eff_BPS_M50 = [sum(mean(h_1.^2, dims = 1)) / sum(mean(h_1_single.^2, dims = 1)),
sum(mean(h_1.^2, dims = 1)) / sum(mean(h_burn10_single.^2, dims = 1)),
sum(mean(h_1.^2, dims = 1)) / sum(mean(h_burn20_single.^2, dims = 1)),
sum(mean(h_1.^2, dims = 1)) / sum(mean(h_burn50_single.^2, dims = 1))]

print("eff_BPS_M50")
print(eff_BPS_M50)

BPS_results = [eff_BPS_M10'; eff_BPS_M20'; eff_BPS_M50']

# 3×4 Matrix{Float64}:
#  0.153434  2.45066  2.20006  1.3797
#  0.194911  1.62799  1.44405  0.905264
#  0.267325  1.11595  0.98803  0.616745

##################################################################################################################
# BOOM
Δt = 10.0; λ = 1.0; ΔM = 10
sampler = BOOM_coupling(∇U, H -Σ_inv, Σ, μ, Δt, ΔM, λ,mn)
discrete_sampler = DiscretePDMPCoupling(sampler)

## Experiment
sim_runs = 1:1000
K = 2
M = 10*K
comp = Array{Float64}(undef, length(sim_runs), 4) # kernel, grad
h_1 = Array{Float64}(undef, length(sim_runs), p)
h_2 = Array{Float64}(undef, length(sim_runs), p)

Random.seed!(0)
for l in sim_runs
    get_estimate!(view(h_1, l, :), view(h_2, l, :),
            view(comp, l, :), discrete_sampler, K, M)
end
#print(quantile(comp[:,1,1],0.9)) = 2 with seed 0 using 100  runs

num_kern = 2 * (comp[:,1,1] .- 1) + max.(1,M + 1 .-comp[:,1,1])

comp_single = Array{Float64}(undef, length(sim_runs), 4) # kernel, grad
h_1_single = Array{Float64}(undef, length(sim_runs), p)
h_2_single = Array{Float64}(undef, length(sim_runs), p)

h_burn10_single = Array{Float64}(undef, length(sim_runs), p)
h_burn20_single = Array{Float64}(undef, length(sim_runs), p)
h_burn50_single = Array{Float64}(undef, length(sim_runs), p)

Random.seed!(1)
for l in sim_runs
    non_coupled_estimator!(view(h_1_single, l, :), view(h_2_single, l, :), view(h_burn10_single, l, :),view(h_burn20_single, l, :),view(h_burn50_single, l, :),
            view(comp_single, l, :), num_kern[l], discrete_sampler)
end

### Check efficiency per dimension
eff_BOOM10 = [sum(mean(h_1.^2, dims = 1)) / sum(mean(h_1_single.^2, dims = 1)),
sum(mean(h_1.^2, dims = 1)) / sum(mean(h_burn10_single.^2, dims = 1)),
sum(mean(h_1.^2, dims = 1)) / sum(mean(h_burn20_single.^2, dims = 1)),
sum(mean(h_1.^2, dims = 1)) / sum(mean(h_burn50_single.^2, dims = 1))]


## Experiment
M = 20*K
comp = Array{Float64}(undef, length(sim_runs), 4) # kernel, grad
h_1 = Array{Float64}(undef, length(sim_runs), p)
h_2 = Array{Float64}(undef, length(sim_runs), p)

Random.seed!(1)
for l in sim_runs
    get_estimate!(view(h_1, l, :), view(h_2, l, :),
            view(comp, l, :), discrete_sampler, K, M)
end

num_kern = 2 * (comp[:,1,1] .- 1) + max.(1,M + 1 .-comp[:,1,1])

comp_single = Array{Float64}(undef, length(sim_runs), 4) # kernel, grad
h_1_single = Array{Float64}(undef, length(sim_runs), p)
h_2_single = Array{Float64}(undef, length(sim_runs), p)

h_burn10_single = Array{Float64}(undef, length(sim_runs), p)
h_burn20_single = Array{Float64}(undef, length(sim_runs), p)
h_burn50_single = Array{Float64}(undef, length(sim_runs), p)

Random.seed!(1)
for l in sim_runs
    non_coupled_estimator!(view(h_1_single, l, :), view(h_2_single, l, :), view(h_burn10_single, l, :),view(h_burn20_single, l, :),view(h_burn50_single, l, :),
            view(comp_single, l, :), num_kern[l], discrete_sampler)
end

### Check efficiency per dimension
eff_BOOM20 = [sum(mean(h_1.^2, dims = 1)) / sum(mean(h_1_single.^2, dims = 1)),
sum(mean(h_1.^2, dims = 1)) / sum(mean(h_burn10_single.^2, dims = 1)),
sum(mean(h_1.^2, dims = 1)) / sum(mean(h_burn20_single.^2, dims = 1)),
sum(mean(h_1.^2, dims = 1)) / sum(mean(h_burn50_single.^2, dims = 1))]


## Experiment
M = 50*K
comp = Array{Float64}(undef, length(sim_runs), 4) # kernel, grad
h_1 = Array{Float64}(undef, length(sim_runs), p)
h_2 = Array{Float64}(undef, length(sim_runs), p)

Random.seed!(1)
for l in sim_runs
    get_estimate!(view(h_1, l, :), view(h_2, l, :),
            view(comp, l, :), discrete_sampler, K, M)
end
#print(quantile(comp[:,1,1],0.9)) = 21 with seed 0 using 100  runs

num_kern = 2 * (comp[:,1,1] .- 1) + max.(1,M + 1 .-comp[:,1,1])

comp_single = Array{Float64}(undef, length(sim_runs), 4) # kernel, grad
h_1_single = Array{Float64}(undef, length(sim_runs), p)
h_2_single = Array{Float64}(undef, length(sim_runs), p)

h_burn10_single = Array{Float64}(undef, length(sim_runs), p)
h_burn20_single = Array{Float64}(undef, length(sim_runs), p)
h_burn50_single = Array{Float64}(undef, length(sim_runs), p)

Random.seed!(1)
for l in sim_runs
    non_coupled_estimator!(view(h_1_single, l, :), view(h_2_single, l, :), view(h_burn10_single, l, :),view(h_burn20_single, l, :),view(h_burn50_single, l, :),
            view(comp_single, l, :), num_kern[l], discrete_sampler)
end

### Check efficiency per dimension
eff_BOOM50 = [sum(mean(h_1.^2, dims = 1)) / sum(mean(h_1_single.^2, dims = 1)),
sum(mean(h_1.^2, dims = 1)) / sum(mean(h_burn10_single.^2, dims = 1)),
sum(mean(h_1.^2, dims = 1)) / sum(mean(h_burn20_single.^2, dims = 1)),
sum(mean(h_1.^2, dims = 1)) / sum(mean(h_burn50_single.^2, dims = 1))]

BOOM_results = [eff_BOOM10'; eff_BOOM20'; eff_BOOM50']

# 3×4 Matrix{Float64}:
#  0.75572   0.935832  0.83226   0.524098
#  0.864168  0.94031   0.835513  0.518283
#  0.962234  0.938169  0.836896  0.522495