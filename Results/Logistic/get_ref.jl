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

function mn(x)
    U(x), x
end


function assign_to_global(dict::Dict{String, Array{Float64}})
    for (key, value) in dict
        @eval global $(Symbol(key)) = $(value)
    end
end

## Obtain the computation to run the long reference chains for

@load "Results/Logistic/raw_results/BPS_K37_l4.0_t8.0.jld2"
K = 37
M_vals = [K, 10*K, 30*K]
assign_to_global(matrices_dict)
print(keys(matrices_dict))

comp_K = 2 * (comp[:,1,1] .- 1) + max.(1,M_vals[1]+1 .-comp[:,1,1])
comp_10K = 2 * (comp[:,1,2] .- 1) + max.(1,M_vals[2]+1 .-comp[:,1,2])
comp_20K = 2 * (comp[:,1,3] .- 1) + max.(1,M_vals[3]+1 .-comp[:,1,3])

println("COMP")
println(mean(comp_K)," ", mean(comp_10K)," ", mean(comp_20K))
println("\n VARIANCE")
println(sum(var(h_1[:,1],dims =1)) + sum(var(h_2[:,:,1],dims =1)))
println(sum(var(h_1[:,2],dims =1)) + sum(var(h_2[:,:,2],dims =1)))
println(sum(var(h_1[:,3],dims =1)) + sum(var(h_2[:,:,3],dims =1)))

print(quantile(comp[:,1,1], 0.85))

print("\nngrad:",mean(comp[:,4,1] + comp[:,2,1]+ comp[:,3,1]))
print("\nngrad:",mean(comp[:,4,2] + comp[:,2,2]+ comp[:,3,2]))
print("\nngrad:",mean(comp[:,4,3] + comp[:,2,3]+ comp[:,3,3]))

function non_coupled_estimator!(est1_out,est2_out, comp_stored, comp_budget, discrete_sampler)
    x1 = randn(p); x2 = randn(p)
    coupled_state = discrete_sampler.init(x1, x2)
    
    niter_burn = ceil(1000.0/discrete_sampler.kernel.Δt)
    for k in 1:niter_burn
        coupled_state = discrete_sampler.kernel(coupled_state)
    end
    cs = coupled_state.coupled_status
    num_grad = cs.num_grad_1+cs.num_grad_coupled

    h = coupled_state.state_1.h
    for k in 1:comp_budget
        coupled_state = discrete_sampler.kernel(coupled_state)
        h = h .+ coupled_state.state_1.h
    end
    h = h ./ comp_budget

    cs = coupled_state.coupled_status
    comp_stored .= cs.num_grad_1+cs.num_grad_coupled - num_grad

    est1_out .= h[1]
    est2_out .= h[2]
    return Nothing
end

# TODO: Future work to replace Random seed with 
# reproducible random behaviour with Threads for parallel computing

# Run BPS sampler for same compute
Δt = 8.0; λᵣ = 4.0; ΔM = 80
sampler = BPS_coupling(∇U, H_bound, Δt, ΔM, λᵣ, mn)
discrete_sampler = DiscretePDMPCoupling(sampler)

sim_runs = 1:1000
refh_1 = Array{Float64}(undef, length(sim_runs), 3)
refh_2 = Array{Float64}(undef, length(sim_runs), p, 3)
refcomp = Array{Float64}(undef, length(sim_runs), 3)

println("comp 1")
compute_budget = floor(mean(comp_K))
Threads.@threads for l in sim_runs
    non_coupled_estimator!(view(refh_1, l, 1), view(refh_2, l, :, 1), view(refcomp, l, 1),
            compute_budget, discrete_sampler)
end

println("comp 2")
compute_budget = floor(mean(comp_10K))
Threads.@threads for l in sim_runs
    non_coupled_estimator!(view(refh_1, l, 2), view(refh_2, l, :, 2), view(refcomp, l, 2),
            compute_budget, discrete_sampler)
end

println("comp 3")
compute_budget = floor(mean(comp_20K))
Threads.@threads for l in sim_runs
    non_coupled_estimator!(view(refh_1, l, 3), view(refh_2, l, :, 3), view(refcomp, l, 3),
            compute_budget, discrete_sampler)
end
matrices_dict_ref = Dict("refh_1" => refh_1, "refh_2" => refh_2, "refcomp" => refcomp,
                        "BPSParam" => [Δt, λᵣ, K], "comp" => comp)
@save "Results/Logistic/raw_results/BPS_ref.jld2" matrices_dict_ref

println("\n VARIANCE (refrence)")
println(sum(var(refh_1[:,1],dims =1))," ",sum(var(refh_2[:,:,1],dims =1)))
println(sum(var(refh_1[:,2],dims =1))," ",sum(var(refh_2[:,:,2],dims =1)))
println(sum(var(refh_1[:,3],dims =1))," ",sum(var(refh_2[:,:,3],dims =1)))

print(mean(refcomp, dims = 1),"\n")
print(quantile(comp[:,4,1] + comp[:,2,1]+ comp[:,3,1], 0.5)," ",quantile(comp[:,4,2] + comp[:,2,2]+ comp[:,3,2], 0.5)," ",quantile(comp[:,4,3] + comp[:,2,3]+ comp[:,3,3], 0.5))

@load "Results/Logistic/raw_results/BOOM_K10_l3.0_t8.0.jld2"
K = 10
function assign_to_global(dict::Dict{String, Array})
    for (key, value) in dict
        @eval global $(Symbol(key)) = $(value)
    end
end
assign_to_global(matrices_dict)
print(keys(matrices_dict))
print(M_vals)

comp_K = 2 * (comp[:,1,1] .- 1) + max.(1,M_vals[1]+1 .-comp[:,1,1])
comp_10K = 2 * (comp[:,1,2] .- 1) + max.(1,M_vals[2]+1 .-comp[:,1,2])
comp_20K = 2 * (comp[:,1,3] .- 1) + max.(1,M_vals[3]+1 .-comp[:,1,3])

println("COMP")
println(mean(comp_K)," ", mean(comp_10K)," ", mean(comp_20K))
println("\n VARIANCE")
println(sum(var(h_1[:,1],dims =1)) + sum(var(h_2[:,:,1],dims =1)))
println(sum(var(h_1[:,2],dims =1)) + sum(var(h_2[:,:,2],dims =1)))
println(sum(var(h_1[:,3],dims =1)) + sum(var(h_2[:,:,3],dims =1)))

print(quantile(comp[:,1,1], 0.85))

print("\nngrad:",mean(comp[:,4,1] + comp[:,2,1]+ comp[:,3,1]))
print("\nngrad:",mean(comp[:,4,2] + comp[:,2,2]+ comp[:,3,2]))
print("\nngrad:",mean(comp[:,4,3] + comp[:,2,3]+ comp[:,3,3]))


Δt = 8.0
λᵣ = 3.0
ΔM = 80
sampler = BOOM_coupling(∇U, H_bound - Σinv, Σ, x_mode, Δt, ΔM, λᵣ, mn)
discrete_sampler = DiscretePDMPCoupling(sampler)

sim_runs = 1:1000
refh_1 = Array{Float64}(undef, length(sim_runs), 3)
refh_2 = Array{Float64}(undef, length(sim_runs), p, 3)
refcomp = Array{Float64}(undef, length(sim_runs), 3)

println("comp 1")
compute_budget = ceil(mean(comp_K))
Threads.@threads for l in sim_runs
    non_coupled_estimator!(view(refh_1, l, 1), view(refh_2, l, :, 1), view(refcomp, l, 1),
            compute_budget, discrete_sampler)
end

println("comp 2")
compute_budget = floor(mean(comp_10K))
Threads.@threads for l in sim_runs
    non_coupled_estimator!(view(refh_1, l, 2), view(refh_2, l, :, 2), view(refcomp, l, 2),
            compute_budget, discrete_sampler)
end

println("comp 3")
compute_budget = floor(mean(comp_20K))
Threads.@threads for l in sim_runs
    non_coupled_estimator!(view(refh_1, l, 3), view(refh_2, l, :, 3), view(refcomp, l, 3),
            compute_budget, discrete_sampler)
end

matrices_dict_ref = Dict("refh_1" => refh_1, "refh_2" => refh_2, "refcomp" => refcomp,
                        "BOOMParam" => [Δt, λᵣ, K], "comp" => comp)
@save "BOOM_ref.jld2" matrices_dict_ref