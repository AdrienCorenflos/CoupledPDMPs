include("../src/pdmps/PDMP_discrete.jl")
include("../src/coupled_estimator.jl")
using Test
using Plots
using Random
using StatsPlots

Random.seed!(1234)
# Gaussian Example
d = 2
μ = Vector([1., 0.5])
function ∇U(x::Vector)
    x - μ
end

λᵣ = 1.;    Δt = 40.;   ΔM = 1
function h_t(x, v, t)
    res = x * t + v/2 * t^2
    #res = v.^2 * t^3/3 + (x-μ) .* v * t^2 + (x-μ).^2 * t
    res = res/Δt
    return res[1], res[end]
end
H = diagm(fill(1., d))
sampler = BPS_coupling(∇U, H, Δt, ΔM, λᵣ, h_t, true)
coupled_sampler = DiscretePDMPCoupling(sampler)

# Rhe Glynn Estimator Check
K = 1
M = K + 1
function one_estimator!(coupling_time, h_km, i_km)
    x0 = randn(d);        x1 = randn(d)
    coupled_state = coupled_sampler.init(x0, x1)
    τ, h_out, i_out = rhee_glynn(coupled_sampler.kernel, coupled_state, K, M, true)
    h_km .= h_out
    i_km .= i_out
    coupling_time .= τ
    return Nothing
end
N = 10_000
coupling_times = zeros(Float64, N)
h_kms = zeros(Float64, N, 2)
i_kms = zeros(Float64, N, 2)
for l in 1:N
    one_estimator!(view(coupling_times, l), view(h_kms, l, :), view(i_kms, l, :))
end

##################################################################333
function h_d(x)
    return x[1], x[end]
    #return (x[1]-μ[1])^2, (x[end]-μ[end])^2
end

ΔM1 = 1
sampler2 = BPS_coupling(∇U, H, Δt, ΔM1, λᵣ, h_d, false)
coupled_sampler2 = DiscretePDMPCoupling(sampler2)
function one_estimator2!(coupling_time, h_km, i_km)
    x0 = randn(d);        x1 = randn(d)
    coupled_state = coupled_sampler2.init(x0, x1)
    τ, h_out, i_out = rhee_glynn(coupled_sampler2.kernel, coupled_state, K, M, true)
    h_km .= h_out
    i_km .= i_out
    coupling_time .= τ
    return Nothing
end

coupling_times2 = zeros(Float64, N)
h_kms2 = zeros(Float64, N, 2)
i_kms2 = zeros(Float64, N, 2)
for l in 1:N
    one_estimator2!(view(coupling_times2, l), view(h_kms2, l, :), view(i_kms2, l, :))
end

#########################################################

ΔM2 = 10
sampler3 = BPS_coupling(∇U, H, Δt, ΔM2, λᵣ, h_d, false)
coupled_sampler3 = DiscretePDMPCoupling(sampler3)
function one_estimator3!(coupling_time, h_km, i_km)
    x0 = randn(d);        x1 = randn(d)
    coupled_state = coupled_sampler3.init(x0, x1)
    τ, h_out, i_out = rhee_glynn(coupled_sampler3.kernel, coupled_state, K, M, true)
    h_km .= h_out
    i_km .= i_out
    coupling_time .= τ
    return Nothing
end

coupling_times3 = zeros(Float64, N)
h_kms3 = zeros(Float64, N, 2)
i_kms3 = zeros(Float64, N, 2)
for l in 1:N
    one_estimator3!(view(coupling_times3, l), view(h_kms3, l, :), view(i_kms3, l, :))
end

## Check plotting
display(boxplot(h_kms[:, 1], title = "Estimates E[X] Δ = $Δt",label = "CTS"))
display(boxplot!(h_kms3[:, 1], label = "DIS M = $ΔM2"))
display(boxplot!(h_kms2[:, 1], label = "DIS M = $ΔM1"))

display(boxplot(h_kms[:, 1], title = "Estimates V[X] Δ = $Δt",label = "CTS"))
display(boxplot!(h_kms3[:, 1], label = "DIS M = $ΔM2"))
display(boxplot!(h_kms2[:, 1], label = "DIS M = $ΔM1"))

display(boxplot(h_kms[:, 1], title = "Estimates V[X] Δ = $Δt",label = "CTS"))
display(boxplot!(h_kms3[:, 1], label = "DIS M = $ΔM2"))

mean(h_kms, dims=1)
mean(h_kms2, dims=1)
mean(h_kms3, dims=1)

#boxplot(i_kms[:, 1])
println("E[x[1]] Unbiased:",mean(h_kms[:, 1]))
println("E[x[1]] Biased:",mean(i_kms[:, 1]))
println("E[x[end]] Unbiased:",mean(h_kms[:, 2]))
println("E[x[end]] Biased:",mean(i_kms[:, 2]))


#boxplot(i_kms[:, 1])
println("E[x[1]] Unbiased:",mean(h_kms3[:, 1]))
println("E[x[1]] Biased:",mean(i_kms3[:, 1]))
println("E[x[end]] Unbiased:",mean(h_kms3[:, 2]))
println("E[x[end]] Biased:",mean(i_kms3[:, 2]))