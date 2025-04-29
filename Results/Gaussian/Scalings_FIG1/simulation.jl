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

function get_Gaussian(p::Int)
    # Potential 
    function U(x::Vector)
        return x' * x /2.0
    end

    # Grad 
    function ∇U(x::Vector)
        return x
    end

    # Hessian of the potential
    function Hess(d)
        Diagonal(diagm(ones(d)))
    end

    H_bound = Hess(p)

    return U, ∇U, Hess, H_bound
end

function coupled_time!(res_out, discrete_sampler, p)
    x1 = randn(p); x2 = randn(p)
    coupled_state = discrete_sampler.init(x1, x2)
    i = 1
    while true
        coupled_state = discrete_sampler.kernel(coupled_state)
        i = i + 1
        #print(i," ")
        if coupled_state.coupled
            break
        end
    end
    cs = coupled_state.coupled_status
    res_out .= [cs.stoch_time, cs.num_grad_1, cs.num_grad_2, cs.num_grad_coupled, i]
    return Nothing
end


dt_values = [0.4,1, 6]
p_values = collect(10:10:100)
N=500
res_out_avg = zeros(Float64, length(dt_values), 5)
res_out_avg_st = zeros(Float64, length(dt_values), length(p_values))
res_out_avg_comp = zeros(Float64, length(dt_values), length(p_values))

for (p_index, p) in enumerate(p_values)
    μ = zeros(p)
    Σinv = Σ = diagm(fill(1., p))
    U, ∇U, Hess, H_bound = get_Gaussian(p)

    for (dt_index, dt) in enumerate(dt_values)
            
        sampler = BPS_coupling(∇U, H_bound, dt, 1, 1.0)
        discrete_sampler = DiscretePDMPCoupling(sampler)

        # Repeat and store grad events
        print("\ndt=",dt,"\n")
        res_out = zeros(Float64, N, 5)
        for l in 1:N
            print( l," ")
            Random.seed!(l)
            coupled_time!(view(res_out, l, :), discrete_sampler, p)
        end
        res_out_avg[dt_index,:] = mean(res_out, dims = 1)
        res_out_avg_st[dt_index, p_index] = res_out_avg[dt_index,1] - dt
        res_out_avg_comp[dt_index, p_index] = res_out_avg[dt_index,2] + res_out_avg[dt_index,3]
    end
    print("\n \n",res_out_avg_st)
end

using Plots
p_BPS = Plots.plot(p_values, res_out_avg_st[1,:] |> Vector,
    xlabel="Dimension (d)", ylabel="Avg Couple Time", 
    label ="Δ = $(dt_values[1])", title = "BPS")

for i in 2:length(dt_values)
    Plots.plot!(p_values, res_out_avg_st[i,:] |> Vector,
    xlabel="Dimension (d)", ylabel="Avg Couple Time", label ="Δ = $(dt_values[i])")
end
display(p_BPS)

cp_BPS = Plots.plot(p_values, res_out_avg_comp[1,:] |> Vector,
    xlabel="Dimension (d)", ylabel="Avg Computation", # title = "BPS",
    label ="Δ = $(dt_values[1])")

for i in 2:length(dt_values)
    Plots.plot!(p_values, res_out_avg_comp[i,:] |> Vector,
    xlabel="Dimension (d)", ylabel="Avg Computation", label ="Δ = $(dt_values[i])")
end
display(cp_BPS)

dt_values = [4,2*pi, 20]
p_values = collect(10:10:100)

res_out_avg = zeros(Float64, length(dt_values), 5)
res_out_avg_st = zeros(Float64, length(dt_values), length(p_values))
res_out_avg_comp = zeros(Float64, length(dt_values), length(p_values))

for (p_index, p) in enumerate(p_values)
    μ = zeros(p)
    Σinv = Σ = diagm(fill(1., p))
    U, ∇U, Hess, H_bound = get_Gaussian(p)

    for (dt_index, dt) in enumerate(dt_values)
            
        sampler = COORD_coupling(∇U, ones(p), p*dt, 1, 1.0)
        discrete_sampler = DiscretePDMPCoupling(sampler)

        # Repeat and store grad events
        print("\ndt=",dt,"\n")
        res_out = zeros(Float64, N, 5)
        for l in 1:N
            print( l," ")
            Random.seed!(l)
            coupled_time!(view(res_out, l, :), discrete_sampler, p)
        end
        res_out_avg[dt_index,:] = mean(res_out, dims = 1)
        res_out_avg_st[dt_index, p_index] = res_out_avg[dt_index,1] - dt
        res_out_avg_comp[dt_index, p_index] = res_out_avg[dt_index,2] + res_out_avg[dt_index,3]
    end
    print("\n \n",res_out_avg_st)
end

using Plots
p_COORD = Plots.plot(p_values, res_out_avg_st[1,:] |> Vector,
    xlabel="Dimension (d)", ylabel="Avg Couple Time", 
    label ="Δ = d × $(dt_values[1])", title = "Coordinate Sampler")

    Plots.plot!(p_values, res_out_avg_st[2,:] |> Vector,
    xlabel="Dimension (d)", ylabel="Avg Couple Time", label ="Δ = d × 2π")

for i in 3:length(dt_values)
    Plots.plot!(p_values, res_out_avg_st[i,:] |> Vector,
    xlabel="Dimension (d)", ylabel="Avg Couple Time", label ="Δ = d × $(dt_values[i])")
end
display(p_COORD)

res_out_avg_comp_sc = res_out_avg_comp * diagm( 1 ./ p_values)

cp_COORD = Plots.plot(p_values, res_out_avg_comp_sc[1,:] |> Vector,
    xlabel="Dimension (d)", ylabel="Avg Computation", #title = "Coordinate Sampler", ylims=[160,300],
    label ="Δ = d × $(dt_values[1])",  legend=:bottomright)

    Plots.plot!(p_values, res_out_avg_comp_sc[2,:] |> Vector,
    xlabel="Dimension (d)", ylabel="Avg Computation", label ="Δ = d × 2π")

for i in 3:length(dt_values)
    Plots.plot!(p_values, res_out_avg_comp_sc[i,:] |> Vector,
    xlabel="Dimension (d)", ylabel="Avg Computation", label ="Δ = d × $(dt_values[i])")
end
display(cp_COORD)

dt_values = [1., 10., 20.]
p_values = collect(1000:1000:9000)

res_out_avg = zeros(Float64, length(dt_values), 5)
res_out_avg_st = zeros(Float64, length(dt_values), length(p_values))
res_out_avg_comp = zeros(Float64, length(dt_values), length(p_values))

for (p_index, p) in enumerate(p_values)
    μ = zeros(p)
    Σinv = Σ = Diagonal(diagm(fill(1., p)))
    U, ∇U, Hess, H_bound = get_Gaussian(p)

    for (dt_index, dt) in enumerate(dt_values)
            
        sampler = BOOM_coupling(∇U, H_bound - Σinv, Σ, μ, dt, 1, 1.0)
        discrete_sampler = DiscretePDMPCoupling(sampler)

        discrete_sampler = DiscretePDMPCoupling(sampler)

        # Repeat and store grad events
        print("\ndt=",dt,"\n")
        res_out = zeros(Float64, N, 5)
        for l in 1:N
            print( l," ")
            Random.seed!(l)
            coupled_time!(view(res_out, l, :), discrete_sampler, p)
        end
        res_out_avg[dt_index,:] = mean(res_out, dims = 1)
        res_out_avg_st[dt_index, p_index] = res_out_avg[dt_index,1] - dt
        res_out_avg_comp[dt_index, p_index] = res_out_avg[dt_index,2] + res_out_avg[dt_index,3]
    end
    print("\n \n",res_out_avg_st)
end

p_BOOM = Plots.plot(p_values, res_out_avg_st[1,:] |> Vector,
    xlabel="Dimension (d)", ylabel="Avg Couple Time", 
    label ="Δ = $(dt_values[1])", title = "Boomerang")

for i in 2:length(dt_values)
    Plots.plot!(p_values, res_out_avg_st[i,:] |> Vector,ylims=[6,21],
    xlabel="Dimension (d)", ylabel="Avg Couple Time", label ="Δ = $(dt_values[i])")
end
display(p_BOOM)

cp_BOOM = Plots.plot(p_values, res_out_avg_comp[1,:] |> Vector,
    xlabel="Dimension (d)", ylabel="Avg Computation", ylims=[32,53],
    label ="Δ = $(dt_values[1])", #title = "Boomerang",
     legend=:topleft)

for i in 2:length(dt_values)
    Plots.plot!(p_values, res_out_avg_comp[i,:] |> Vector,#ylims=[6,21],
    xlabel="Dimension (d)", ylabel="Avg Computation", label ="Δ = $(dt_values[i])")
end
display(cp_BOOM)

l = @layout [a b c]
plot(p_BPS, p_COORD, p_BOOM, layout = l)
plot!(size=(800,250), margin=5Plots.mm)

l = @layout [a b c; d e f]
plot(p_BPS, p_COORD, p_BOOM, cp_BPS, cp_COORD, cp_BOOM, layout = l)
plot!(size=(800,500), margin=5Plots.mm)
