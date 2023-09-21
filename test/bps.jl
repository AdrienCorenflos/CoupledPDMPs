include("../src/pdmps/PDMP_discrete.jl")
include("../src/coupled_estimator.jl")
using Test
using Plots
using Random
using StatsPlots

@testset "Test if BPS kernel samples Gaussian correctly." begin
    Random.seed!(1234)
    # Gaussian Example
    function ∇U(x::Vector)
        x
    end
    d = 2
    H = diagm(fill(1., d))
    λᵣ = 1.;    Δt = 2.;    ΔM=1
    sampler = BPS_coupling(∇U, H, Δt, ΔM, λᵣ)
    state, _, _, _, _ = sampler.init(randn(d), randn(d))
    
    N = 100_000
    states = Matrix{Float64}(undef, N, 2)
    for i in 1:N
        T_seq = (state[1] + Δt):Δt:(state[1] + Δt)
        state, _, _ = sampler.kernel(state, T_seq, 0.0) 
        states[i,:] = state[2]
    end
    #marginalkde(states[:,1], states[:,2])

    @test mean(states[:,1]) ≈ 0. atol = 1e-2 rtol = 1e-2
    @test mean(states[:,2]) ≈ 0. atol = 1e-2 rtol = 1e-2
    @test std(states[:,1]) ≈ 1. atol = 1e-2 rtol = 1e-2
    @test std(states[:,2]) ≈ 1. atol = 1e-2 rtol = 1e-2
end

@testset "Test if BPS coupled kernel samples Gaussian correctly." begin
    Random.seed!(1234)
    # Gaussian Example
    d = 2
    μ = Vector([1., 0.5])
    function ∇U(x::Vector)
        x - μ
    end
    function h(x)
        return x[1], x[1]^2, x[end], x[end]^2 
    end
    H = diagm(fill(1., d))
    λᵣ = 1.;    Δt = 1.; ΔM = 1
    sampler = BPS_coupling(∇U, H, Δt, ΔM, λᵣ, h, false)
    coupled_sampler = DiscretePDMPCoupling(sampler)
    
    # Rhe Glynn Estimator Check
    K = 10
    M = 100
    function one_estimator!(coupling_time, h_km, i_km)
        x0 = randn(d) .+ 10;        x1 = randn(d) .+ 10
        coupled_state = coupled_sampler.init(x0, x1)
        τ, h_out, i_out = rhee_glynn(coupled_sampler.kernel, coupled_state, K, M, true)
        h_km .= h_out
        i_km .= i_out
        coupling_time .= τ
        return Nothing
    end

    N = 50_000
    coupling_times = zeros(Float64, N)
    h_kms = zeros(Float64, N, 4)
    i_kms = zeros(Float64, N, 4)
    for l in 1:N
        one_estimator!(view(coupling_times, l), view(h_kms, l, :), view(i_kms, l, :))
    end
    #boxplot(h_kms[:, 1])
    #boxplot(i_kms[:, 1])

    println("E[x[1]] Unbiased:",mean(h_kms[:, 1]))
    println("E[x[1]] Biased:",mean(i_kms[:, 1]))
    println("E[x[end]] Unbiased:",mean(h_kms[:, 3]))
    println("E[x[end]] Biased:",mean(i_kms[:, 3]))
    println("E[x[1]^2] Unbiased:",mean(h_kms[:, 2]))
    println("E[x[1]^2] Biased:",mean(i_kms[:, 2]))
    println("E[x[end]^2] Unbiased:",mean(h_kms[:, 4]))
    println("E[x[end]^2] Biased:",mean(i_kms[:, 4]))

    tol = 0.05
    @test mean(h_kms[:, 1]) ≈ μ[1] atol = tol
    @test mean(h_kms[:, 3]) ≈ μ[end] atol = tol
    @test mean(h_kms[:, 2]) ≈ μ[1]^2 + 1 atol = tol
    @test mean(h_kms[:, 4]) ≈ μ[end]^2 + 1 atol = tol
end

@testset "Test if BPS kernel samples Logistic." begin
    Random.seed!(1234)
    # Logistic Example
    d = 2
    X = Matrix(randn(100, d))
    param = Vector([1., 2.])
    q = exp.(X*param)
    Y = Vector(rand(100) .< (q./(1 .+ q)))
    prior_precision = 1.
    function ∇U(x::Vector)
        μ = X*x
        η_val = exp.(μ)
        prior_terms = prior_precision * x 
        ϕ = X .* (η_val ./(1 .+ η_val) .- Y)
        grad = sum(ϕ, dims=1)[1,:] + prior_terms
        return grad
    end

    Σ = prior_precision*diagm(fill(1., d))
    H = (X' * X)/4 + Σ

    λᵣ = sqrt(d);    Δt = 1.;   ΔM=1
    sampler = BPS_coupling(∇U, H, Δt, ΔM, λᵣ)
    state, _, _, _, _ = sampler.init(randn(d), randn(d))
    
    N = 10_000
    states = Matrix{Float64}(undef, N, d)
    for i in 1:N
        T_seq = (state[1] + Δt):Δt:(state[1] + Δt)
        state, _, _ = sampler.kernel(state, T_seq, 0.0) 
        states[i,:] = state[2]
    end
    #marginalkde(states[:,1], states[:,2])

    coupled_sampler = DiscretePDMPCoupling(sampler)
    coupled_state = coupled_sampler.init(randn(d), randn(d))

    # Get samples to check vis
    N = 500
    states_1 = Matrix{Float64}(undef, N, d)
    states_2 = Matrix{Float64}(undef, N, d)
    states_1[1,:] = coupled_state.state_1.z[2]
    states_2[1,:] = coupled_state.state_2.z[2]
    for i in 2:N
        coupled_state = coupled_sampler.kernel(coupled_state) 
        states_1[i,:] = coupled_state.state_1.z[2]
        states_2[i,:] = coupled_state.state_2.z[2]
    end
    cpl = coupled_state.coupled
    plot(states_1[:,1], title = "Coupled: $cpl")
    plot!(states_2[:,1])

    plot(states_1[:,1], states_1[:,2])
    plot!(states_2[:,1], states_2[:,2])

end
