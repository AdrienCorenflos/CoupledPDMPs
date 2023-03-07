include("../src/coupled_estimator.jl")
using Distributions
using Random
using Statistics
using StatsBase
using Test
using Distributed
using SharedArrays
using Plots


struct MyState 
    x::Float64
    h::Any
end

struct MyCoupledState
    state_1::MyState
    state_2::MyState
    coupled::Bool
end

function h(x)
    return x, x^2 - 1, x^3
end

function kernel(state, ρ::Float64=0.99)
    # The kernel is a simple AR(1) process with N(0, 1) as a stationary distribution.

    α = √(1 - ρ^2)
    x = state.x

    new_x = ρ * x + α * randn()
    new_h = h(new_x)

    return MyState(new_x, new_h)
end


function reflection_maximal(μ₁, μ₂, σ)
    # The reflection maximal coupling between two 1D distributions.

    z = (μ₁ - μ₂) / σ

    ϵ = randn()
    logᵤ = log(rand())

    z += ϵ

    ℓₑ = -ϵ^2 / 2
    ℓₜ = -z^2 / 2

    coupled = logᵤ < ℓₜ - ℓₑ

    z = coupled ? z : -ϵ

    x = μ₁ + σ * ϵ
    y = μ₂ + σ * z
    return x, y, coupled
end


function coupled_kernel(coupled_state, ρ::Float64=0.99)
    # The kernel is a simple AR(1) process with N(0, 1) as a stationary distribution.
    state_1, state_2 = coupled_state.state_1, coupled_state.state_2
    α = √(1 - ρ^2)
    x₁, x₂ = state_1.x, state_2.x
    
    # means of the two distributions
    μ₁, μ₂ = ρ * x₁, ρ * x₂

    new_x₁, new_x₂, coupled = reflection_maximal(μ₁, μ₂, α)
    new_h₁, new_h₂ = h(new_x₁), h(new_x₂)
    new_state_1 = MyState(new_x₁, new_h₁)
    new_state_2 = MyState(new_x₂, new_h₂)

    new_coupled_state = MyCoupledState(new_state_1, new_state_2, coupled)
    return new_coupled_state
end

@testset "Rhee and Glynn for simple ergodic sequence." begin
    Random.seed!(1234)
    # This test is pretty long to run, but we need to test convergence of the bias corrected estimator and lackthereof for the standard estimator.
    K = 250
    M = 2_500
    function one_estimator!(coupling_time, h_km, i_km)
        x₁, x₂ = randn(Float64) + 10, randn(Float64) + 10  # Create a bias by starting largely too high compated to stationarity.
        state₁ = MyState(x₁, h(x₁))
        state₂ = MyState(x₂, h(x₂))
    
        state₁ = kernel(state₁)
        coupled_state = MyCoupledState(state₁, state₂, false)
    
    
        τ, h_out, i_out = rhee_glynn(coupled_kernel, coupled_state, K, M, true)
        h_km .= h_out
        i_km .= i_out
        coupling_time .= τ
        return Nothing
    end

    L = 100_000
    coupling_times = zeros(Float64, L)
    h_kms = zeros(Float64, L, 3)
    i_kms = zeros(Float64, L, 3)
    for l in 1:L
        one_estimator!(view(coupling_times, l), view(h_kms, l, :), view(i_kms, l, :))
    end

    # println("")
    # show(summarystats(h_kms[:, 1]))    
    # println("")
    # show(summarystats(coupling_times))

    # println("")
    # show(mean(h_kms, dims=1))
    # println("")
    # show(mean(i_kms, dims=1))

    @test mean(h_kms[:, 1]) ≈ 0. atol=0.01
    @test mean(h_kms[:, 2]) ≈ 0. atol=0.01
    @test mean(h_kms[:, 3]) ≈ 0. atol=0.1
    @test mean(i_kms[:, 1]) ≈ 0.04 atol=0.01
    @test mean(i_kms[:, 2]) ≈ 0.01 atol=0.01
    @test mean(i_kms[:, 3]) ≈ 0.12 atol=0.01



end