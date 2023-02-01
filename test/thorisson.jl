include("../src/generic_couplings/thorisson.jl")
using Random
using Statistics
using Test

@testset "Check that Thorisson's coupling works for 1D distributions." begin
    x₁, x₂ = randn(Float64), randn(Float64)

    d₁, d₂ = Normal(x₁), Normal(x₂)
    N = 500_000

    res₁ = zeros(Float64, N)
    res₂ = zeros(Float64, N)
    coupled = zeros(Bool, N)
    for i = 1:N
        res₁[i], res₂[i], coupled[i] = thorisson(d₁, d₂)
    end

    @test mean(res₁) ≈ x₁ atol = 1e-2 rtol = 1e-2
    @test mean(res₂) ≈ x₂ atol = 1e-2 rtol = 1e-2
    @test mean(coupled) > 0
    @test all(res₁[coupled] .== res₂[coupled])
    @test ~all(res₁[.~coupled] .== res₂[.~coupled])


end