include("../src/poisson/affine.jl")
using Random
using Statistics
using Test
using QuadGK
using KernelDensity
using Plots



@testset "Constant affine rate is homogeneous." begin
    Random.seed!(1234)
    a, b, c = 0., 0.1, 0.3
    d = AffinePoisson(a, b, c)
    N = 100_000
    res = zeros(Float64, N)
    rand!(d,res)
    @test mean(res) ≈ 1 / (b + c) atol = 1e-2 rtol = 1e-2
    @test std(res) ≈ 1 / (b + c) atol = 1e-2 rtol = 1e-2
    
end


function test_one(a, b, c, res)
    d = AffinePoisson(a, b, c)
    rand!(d,res)
    kde_max = sort(res)[length(res)-100]
    kde_res = kde(res, boundary=(0, kde_max),  npoints = 2^15) 
    x_vals = kde_res.x
    est_lpdf = log.(kde_res.density)

    # plot(x_vals, est_lpdf, label = "Estimated")
    # plot!(x_vals, logpdf.(d, x_vals), label = "True")
    # savefig("affine $a $b $c.png")

    @testset "Pdf integrates to 1: $a, $b, $c" begin
        int = quadgk(x -> exp(logpdf(d, x)), 0, Inf)[1]
        @test int ≈ 1
    end

    @testset "Pdf and rand match: $a, $b, $c" begin
        @test mean(abs.(est_lpdf - logpdf.(d, x_vals))) ≈ 0 atol = 5e-1 
    end
end

@testset "Affine Poisson test" begin
    Random.seed!(42)
    N = 1_000_000
    res = zeros(Float64, N)

    a, b, c, shift = 1., 0.5, 0.1, -1.
    test_one(a, b, c, shift, res)
    a, b, c, shift = 1., -0.5, .1, -1.
    test_one(a, b, c, shift, res)

    a, b, c, shift = 3., -2., 1., 1.
    test_one(a, b, c, shift, res)
    a, b, c, shift = -3., 2., 2., 1.
    test_one(a, b, c, shift, res)

    a, b, c, shift = -1., -0.5, 0.1, 1.
    test_one(a, b, c, shift, res)

    a, b, c, shift = 0., 0.5, 0.1, 1.
    test_one(a, b, c, shift, res)

    a, b, c, shift = 0., -0.5, 0.1, 1.
    test_one(a, b, c, shift, res)

    a, b, c, shift = 0., 0., 0.1, 1.
    test_one(a, b, c, shift, res)
end
