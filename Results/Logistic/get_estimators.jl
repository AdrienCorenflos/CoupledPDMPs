## Include the packages and samplers
using Distributions
using Random
using LinearAlgebra
using StatsBase
using Base.Threads
using JLD2
using CSV
using DataFrames
include("../../src/pdmps/PDMP_discrete.jl")
include("../../src/coupled_estimator_verb.jl")

## Generate data, gradient and potential function
csv_file_path = "Results/Logistic/small_logit_data.csv"

# Load the CSV file into a DataFrame
data = CSV.File(csv_file_path) |> DataFrame
Y = Vector(data[:,:y])
X = Matrix(data[:, 1:(end-1)])

p = size(X)[2]
prior_precision = ones(p)

# Potential 
function U(x::Vector)
    μ = X*x
    η_val = exp.(μ)
    return (sum(log.(1.0 .+ η_val) -Y .* μ) +  x' * (prior_precision .* x)/2.0)
end
# Grad 
function ∇U(x::Vector)
    μ = X*x
    η_val = exp.(μ)
    prior_terms = prior_precision .* x 
    ϕ = X .* (η_val ./(1 .+ η_val) .- Y)
    grad = sum(ϕ, dims=1)[1,:] + prior_terms
    return grad
end
# Hessian of the potential
function Hess(x)
    μ = X*x
    η_val = exp.(μ)
    sum_term = sum(X[i,:] * X[i,:]' * η_val[i] / (1.0 + η_val[i])^2 for i in 1:length(η_val))
    diagm(fill(1., p)) .* prior_precision + Symmetric(sum_term, :U)
end
H_bound = (X' * X)/4 + diagm(fill(1., p)).*prior_precision;

using Optim

∇U! = function(storage, x)
    storage[:] = ∇U(x)
end

res = Optim.optimize(U, ∇U!, randn(p), LBFGS(), Optim.Options(g_tol = 1e-15, show_trace = false))
x_mode = Optim.minimizer(res)

Σinv = Hess(x_mode) 
Σ = inv(factorize(Σinv))
Σinv = inv(Σ); 

function get_estimate!(est1_out,est2_out, comp_out, discrete_sampler, K, M)
    x1 = randn(p); x2 = randn(p)
    coupled_state = discrete_sampler.init(x1, x2)
    
    (kern, (est1, est2), _, cs) = rhee_glynn(discrete_sampler.kernel, coupled_state, K, M)

    est1_out .= est1
    est2_out .= est2

    comp_out .= [kern, cs.num_grad_1, cs.num_grad_2, cs.num_grad_coupled]
    return Nothing
end

function mn(x)
    U(x), x
end

print(Threads.nthreads())
Random.seed!(1)
sim_runs = 1:1000
K = 37
M_vals = [K, 10*K, 30*K]

comp = Array{Float64}(undef, length(sim_runs), 4, 3) # kernel, grad
h_1 = Array{Float64}(undef, length(sim_runs), 3)
h_2 = Array{Float64}(undef, length(sim_runs), p, 3)

Δt = 8.0
λᵣ = 4.0
ΔM = 80
sampler = BPS_coupling(∇U, H_bound, Δt, ΔM, λᵣ, mn)
discrete_sampler = DiscretePDMPCoupling(sampler)

for (m_ind, m) in enumerate(M_vals)
    println("At m=",m)
    for l in sim_runs
        get_estimate!(view(h_1, l, m_ind), view(h_2, l, :, m_ind), 
                view(comp, l, :, m_ind), discrete_sampler, K, m)
    end
    matrices_dict = Dict("h_1" => h_1, "h_2" => h_2,  
                        "comp" => comp, "tuning" => [Δt, ΔM, λᵣ, K])
    
    @save "BPS_K$(K)_l$(λᵣ)_t$(Δt).jld2" matrices_dict
end

Random.seed!(1)
sim_runs = 1:1000
K = 10
M_vals = [K, 50*K, 100*K]

comp = Array{Float64}(undef, length(sim_runs), 4, 3) # kernel, grad
h_1 = Array{Float64}(undef, length(sim_runs), 3)
h_2 = Array{Float64}(undef, length(sim_runs), p, 3)

Δt = 8.0
λᵣ = 3.0
ΔM = 80
sampler = BOOM_coupling(∇U, H_bound - Σinv, Σ, x_mode, Δt, ΔM, λᵣ, mn)
discrete_sampler = DiscretePDMPCoupling(sampler)

for (m_ind, m) in enumerate(M_vals)
    println("At m=",m)
    for l in sim_runs
        get_estimate!(view(h_1, l, m_ind), view(h_2, l, :, m_ind),  
                view(comp, l, :, m_ind), discrete_sampler, K, m)
    end
    matrices_dict = Dict("h_1" => h_1, "h_2" => h_2, "h_3" => h_3, "M_vals" => M_vals,
                        "comp" => comp, "tuning" => [Δt, ΔM, λᵣ, K])
    
    @save "BOOM_K$(K)_l$(λᵣ)_t$(Δt).jld2" matrices_dict
end

