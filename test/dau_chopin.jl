include("../src/generic_couplings/dau_chopin.jl")
include("../src/poisson/affine.jl")
using Random
using Statistics
using Test



@testset "Check that Dang's coupling works for 1D distributions." begin
    x₁, x₂ = randn(Float64), randn(Float64)

    d₁, d₂ = Normal(x₁), Normal(x₂)
    N = 500_000

    res₁ = zeros(Float64, N)
    res₂ = zeros(Float64, N)
    coupled = zeros(Bool, N)

    Γ() = rand(d₁), rand(d₂)
    for i = 1:N
        res₁[i], res₂[i], coupled[i] = dau_chopin(d₁, d₂, Γ)
    end

    @test mean(res₁) ≈ x₁ atol = 1e-2 rtol = 1e-2
    @test mean(res₂) ≈ x₂ atol = 1e-2 rtol = 1e-2
    @test mean(coupled) > 0
    @test all(res₁[coupled] .== res₂[coupled])
    @test ~all(res₁[.~coupled] .== res₂[.~coupled])
end