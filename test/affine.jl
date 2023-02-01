include("../src/poisson/affine.jl")
using Random
using Statistics
using Test

@testset "Check that the affine Poisson with constant rate is an homogeneous one." begin
    a = Float64(0.0)
    b = 0.1
    c = Float64(0.0)
    d = AffinePoisson(a, b, c)
    N = 50
    res = zeros(Float64, N)
    rand!(d,res)
    show(mean(res))
    @test mean(res) ≈ 1 / b atol = 1e-2 rtol = 1e-2
    @test std(res) ≈ 1 / b atol = 1e-2 rtol = 1e-2
    
end


@testset "Check that the affine Poisson pdf and rand match." begin
    @test error("Not implemented yet")
end