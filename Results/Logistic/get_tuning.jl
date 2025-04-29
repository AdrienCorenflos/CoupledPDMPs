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
p = size(X)[2]

function get_coupling!(est1_out,est2_out, comp_out, discrete_sampler)
    x1 = randn(p); x2 = randn(p)
    coupled_state = discrete_sampler.init(x1, x2)
    i = 1
    while true
        coupled_state = discrete_sampler.kernel(coupled_state)
        i = i + 1
        if coupled_state.coupled
            break
        end
    end
    niter_burn = ceil(max(1000.0 - coupled_state.state_1.z[1], 1.0)/discrete_sampler.kernel.Δt)
    for k in 1:niter_burn
        coupled_state = discrete_sampler.kernel(coupled_state)
    end

    cs = coupled_state.coupled_status
    num_grad = cs.num_grad_1 + cs.num_grad_coupled

    niter = ceil(1000.0/discrete_sampler.kernel.Δt)
    h = coupled_state.state_1.h
    for k in 1:niter
        coupled_state = discrete_sampler.kernel(coupled_state)
        h = h .+ coupled_state.state_1.h
    end
    h = h ./ niter

    est1_out .= h[1]
    est2_out .= h[2]

    cs = coupled_state.coupled_status
    comp_out .= [i, cs.num_grad_1+cs.num_grad_2, cs.num_grad_1 + cs.num_grad_coupled - num_grad, cs.stoch_time]
    return Nothing
end

# TODO: Future work to replace Random seed with 
# reproducible random behaviour with Threads for parallel computing

Random.seed!(0) 
dt_values = [1., 2., 4., 8]
l_values = [1, 2, 3, 4.]

var_h1 = zeros(Float64, length(dt_values), length(l_values))
var_h2 = zeros(Float64, length(dt_values), length(l_values))
Kern = zeros(Float64, length(dt_values), length(l_values))
Kern95 = zeros(Float64, length(dt_values), length(l_values))
nGradCouple = zeros(Float64, length(dt_values), length(l_values))
nGradTotal = zeros(Float64, length(dt_values), length(l_values))
stochtCouple = zeros(Float64, length(dt_values), length(l_values))
stochtCouple95 = zeros(Float64, length(dt_values), length(l_values))
n_tune_reps = 100
check = zeros(Float64, length(dt_values), length(l_values))

for (l_index, l) in enumerate(l_values)
    for (dt_index, dt) in enumerate(dt_values)
                
        sampler = BPS_coupling(∇U, H_bound, dt, Int(dt*10), l, mn)
        discrete_sampler = DiscretePDMPCoupling(sampler)

        # Repeat and store grad events
        print("\ndt=",dt,"\n")
        h1s = zeros(Float64, n_tune_reps)
        h2s = zeros(Float64, n_tune_reps, p)
        comp = zeros(Float64, n_tune_reps, 4)

        a = zeros(Float64, n_tune_reps)
        Threads.@threads for l in 1:n_tune_reps
            a[l] = rand()
            get_coupling!(view(h1s, l),view(h2s, l, :), view(comp, l, :),
            discrete_sampler)
        end
        check[dt_index, l_index] = length(unique(a))
        var_h1[dt_index, l_index] = var(h1s)
        var_h2[dt_index, l_index] = sum(var(h2s, dims = 1))
        Kern[dt_index, l_index] = mean(comp[:,1])
        Kern95[dt_index, l_index] = StatsBase.quantile(comp[:,1], 0.95)
        nGradCouple[dt_index, l_index] = mean(comp[:,2])
        nGradTotal[dt_index, l_index] = mean(comp[:,3])
        stochtCouple[dt_index, l_index] = mean(comp[:,4])
        stochtCouple95[dt_index, l_index] = StatsBase.quantile(comp[:,4], 0.95)
    end
    print("\n \n",nGradTotal .* (var_h1 + var_h2)," ", check[:, l_index])
end


matrices_dict_tuning = Dict("var_h1" => var_h1, "var_h2" => var_h2, "stochtCouple95" => stochtCouple95, 
                        "Kern" => Kern, "nGradCouple" => nGradCouple, "nGradTotal" => nGradTotal, 
                        "dt_values" => dt_values, "l_values" => l_values)
@save "BPS_tuning.jld2" matrices_dict_tuning

# TODO: Future work to replace Random seed with 
# reproducible random behaviour with Threads for parallel computing

Random.seed!(0)
dt_values = [1., 2., 4., 8]
l_values = [0.5, 1, 2, 3, 4.]

var_h1 = zeros(Float64, length(dt_values), length(l_values))
var_h2 = zeros(Float64, length(dt_values), length(l_values))
Kern = zeros(Float64, length(dt_values), length(l_values))
Kern95 = zeros(Float64, length(dt_values), length(l_values))
nGradCouple = zeros(Float64, length(dt_values), length(l_values))
nGradTotal = zeros(Float64, length(dt_values), length(l_values))
stochtCouple = zeros(Float64, length(dt_values), length(l_values))
stochtCouple95 = zeros(Float64, length(dt_values), length(l_values))
n_tune_reps = 100
check = zeros(Float64, length(dt_values), length(l_values))

for (l_index, l) in enumerate(l_values)
    for (dt_index, dt) in enumerate(dt_values)
                
        sampler = BOOM_coupling(∇U, H_bound - Σinv, Σ, x_mode, dt, Int(dt*10), l, mn)
        discrete_sampler = DiscretePDMPCoupling(sampler)

        # Repeat and store grad events
        print("\ndt=",dt,"\n")
        h1s = zeros(Float64, n_tune_reps)
        h2s = zeros(Float64, n_tune_reps, p)
        comp = zeros(Float64, n_tune_reps, 4)

        a = zeros(Float64, n_tune_reps)
        Threads.@threads for l in 1:n_tune_reps
            a[l] = rand()
            get_coupling!(view(h1s, l),view(h2s, l, :), view(comp, l, :),
            discrete_sampler)
        end
        check[dt_index, l_index] = length(unique(a))
        var_h1[dt_index, l_index] = var(h1s)
        var_h2[dt_index, l_index] = sum(var(h2s, dims = 1))
        Kern[dt_index, l_index] = mean(comp[:,1])
        Kern95[dt_index, l_index] = StatsBase.quantile(comp[:,1], 0.95)
        nGradCouple[dt_index, l_index] = mean(comp[:,2])
        nGradTotal[dt_index, l_index] = mean(comp[:,3])
        stochtCouple[dt_index, l_index] = mean(comp[:,4])
        stochtCouple95[dt_index, l_index] = StatsBase.quantile(comp[:,4], 0.95)
    end
    print("\n \n",nGradTotal .* (var_h1 + var_h2)," ", check[:, l_index])
end


matrices_dict_tuning = Dict("var_h1" => var_h1, "var_h2" => var_h2, "stochtCouple95" => stochtCouple95, 
                        "Kern" => Kern, "nGradCouple" => nGradCouple, "nGradTotal" => nGradTotal, 
                        "dt_values" => dt_values, "l_values" => l_values)
@save "Results/Logistic/BOOM_tuning.jld2" matrices_dict_tuning





