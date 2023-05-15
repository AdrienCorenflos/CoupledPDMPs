include("../src/coupled_estimator.jl")
include("../src/pdmps/BPS_discrete.jl")

using Distributions
using Random
using StatsBase


using Plots: plot, plot!, scatter, scatter!, histogram
using StatsPlots: boxplot

function ∇U(x::Vector)
    x
end
function couple_time(kernel, coupledstate)
    i=1
    while !coupledstate.coupled
        i+=1
        coupledstate = kernel(coupledstate)
    end
    return i, coupledstate.coupled_time
end
function h(x)
    return x
end

d = 1
diag_d = fill(1., d)
H = diagm(diag_d)
λᵣ = 1.0
Δt = 0.7

couple_mode = "antithetic"
sampler = BPS_coupling(∇U, H, Δt, λᵣ, couple_mode)

hs_all = Matrix{Float64}(undef, 1000, 4)
t_DK_all = Matrix{Int}(undef, 1000, 4)
t_stoch_all = Matrix{Float64}(undef, 1000, 4)

Random.seed!(1)
method = [1 2 3 4]
for m in 1:4
    times = []
    for R in 1:1000
        i=1
        #x1 = randn(dim(H)) .+ 10.0; x2 = randn(dim(H)) .+ 10.0
        x1 = randn(dim(H)); x2 = randn(dim(H)) 
        coupled_state = coupled_init(x1, x2, method[m])
        push!(times, couple_time(coupled_kernel, coupled_state))
    end
    times_DK = [times[i][1] for i in 1:1000]
    times_stoch = [times[i][2].stoch_time for i in 1:1000]
    K = Int(ceil(quantile(times_DK, 0.9)))
    M = K*10
    println("K ",K," M ",M)
    R = 1_000
    couple_info_all = Vector{Any}(undef, R)
    for i in 1:R
        #x1 = randn(dim(H)) .+ 10.0; x2 = randn(dim(H)) .+ 10.0
        x1 = randn(dim(H)); x2 = randn(dim(H)) 
        coupled_state = coupled_init(x1, x2, method[m])
        #τ, h_out, i_out
        couple_info_all[i] =  rhee_glynn(coupled_kernel, coupled_state, K, M, true)
    end
    hs = [couple_info_all[i][2][1][1] for i in 1:R]
    hs_all[:,m] = hs
    t_DK_all[:,m] = times_DK
    t_stoch_all[:,m] = times_stoch
end

ve = boxplot(["M1" "M2" "M3_R3" "M3_R4"], hs_all, ylabel = "E[x]",
                legend = false, title = "Variability of estimator")
dk = boxplot(["M1" "M2" "M3_R3" "M3_R4"],t_DK_all, legend = false, title = "Coupling Times", 
            ylabel = "Number Kernel Updates")
plot(ve,dk)









########################
method = 1
times = []
for R in 1:1000
    i=1
    x1 = randn(dim(H)) .+ 10.0; x2 = randn(dim(H)) .+ 10.0
    coupled_state = coupled_init(x1, x2, method)
    push!(times, couple_time(coupled_kernel, coupled_state))
end
times_DK = [times[i][1] for i in 1:1000]
times_stoch = [times[i][2].stoch_time for i in 1:1000]
K = Int(ceil(quantile(times_DK, 0.9)))
M = K*10
R = 1_000
couple_info_all = Vector{Any}(undef, R)
for i in 1:R
    x1 = randn(dim(H)) .+ 10.0; x2 = randn(dim(H)) .+ 10.0
    coupled_state = coupled_init(x1, x2, 1)
    #τ, h_out, i_out
    couple_info_all[i] =  rhee_glynn(coupled_kernel, coupled_state, K, M, true)
end
hs = [couple_info_all[i][2][1][1] for i in 1:R]
hs_all[:,method] = hs
t_DK_all[:,method] = times_DK
t_stoch_all[:,method] = times_stoch

# K 54 M 540

########################
method = 2
times = []
for R in 1:1000
    i=1
    x1 = randn(dim(H)) .+ 10.0; x2 = randn(dim(H)) .+ 10.0
    coupled_state = coupled_init(x1, x2, method)
    push!(times, couple_time(coupled_kernel, coupled_state))
end
times_DK = [times[i][1] for i in 1:1000]
times_stoch = [times[i][2].stoch_time for i in 1:1000]
K = Int(ceil(quantile(times_DK, 0.99)))
M = K*10
R = 1_000
couple_info_all = Vector{Any}(undef, R)
for i in 1:R
    x1 = randn(dim(H)) .+ 10.0; x2 = randn(dim(H)) .+ 10.0
    coupled_state = coupled_init(x1, x2, method)
    #τ, h_out, i_out
    couple_info_all[i] =  rhee_glynn(coupled_kernel, coupled_state, K, M, true)
end
hs = [couple_info_all[i][2][1][1] for i in 1:R]
hs_all[:,method] = hs
t_DK_all[:,method] = times_DK
t_stoch_all[:,method] = times_stoch




boxplot(hs_all)
boxplot(t_DK_all[:,1:2])



