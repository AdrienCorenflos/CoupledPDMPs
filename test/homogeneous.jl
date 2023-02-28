include("../src/poisson/homogeneous.jl")
using Random
using Statistics
using Test

@testset "Check that the Homogeneous Poisson returns exponentials." begin
    rate = rand(Float64)

    d = HomogeneousPoisson(rate)
    N = 500_000
    res = zeros(Float64, N)
    rand!(d, res)

    @test mean(res) ≈ 1 / rate atol = 1e-2 rtol = 1e-2
    @test std(res) ≈ 1 / rate atol = 1e-2 rtol = 1e-2

end
