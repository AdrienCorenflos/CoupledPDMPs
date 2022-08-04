using CoupledPDMPs
using Random
using Statistics
using Test

@testset "Check that the reflection maximal coupling samples from the right marginals and is maximal." begin
    Random.seed!(123456)
    N, d = 50_000, 3
    x₁, x₂ = randn(d), randn(d)
    σ = 1. + rand()

    res₁ = zeros(Float64, (d, N))
    res₂ = zeros(Float64, (d, N))
    coupled = zeros(Bool, (N,))
    for i = 1:N
        coupled[i] = reflection_maximal!(x₁, x₂, σ, view(res₁, :, i), view(res₂, :, i))
    end

    @test mean(res₁, dims=2) ≈ x₁ atol = 1e-2 rtol = 1e-2
    @test mean(res₂, dims=2) ≈ x₂ atol = 1e-2 rtol = 1e-2
    @test mean(coupled) > 0


end