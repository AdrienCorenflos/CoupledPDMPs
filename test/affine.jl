include("../src/poisson/affine.jl")
using Random
using Statistics
using Test
using KernelDensity
using Plots: plot, plot!

@testset "Check that the affine Poisson with constant rate is an homogeneous one." begin
    a = Float64(0.0)
    b = 0.1
    c = Float64(0.0)
    d = AffinePoisson(a, b, c)
    N = 100_000
    res = zeros(Float64, N)
    rand!(d,res)
    show(mean(res))
    @test mean(res) ≈ 1 / b atol = 1e-2 rtol = 1e-2
    @test std(res) ≈ 1 / b atol = 1e-2 rtol = 1e-2
    
end


@testset "Check that the affine Poisson pdf and rand match." begin
    # Currently tests against empirical pdf but this is a bit to variable to work well...
    a = Float64(.0)
    b = -0.5
    c = Float64(.2)
    d = AffinePoisson(a, b, c)
    N = 100_000
    res = zeros(Float64, N)
    rand!(d,res)
    kde_max = sort(res)[N-100]
    kde_res = kde(res, boundary=(0, kde_max),  npoints = 2^15)
    x_vals = kde_res.x
    est_lpdf = log.(kde_res.density)
    plot(x_vals, est_lpdf)
    plot!(x_vals,logpdf.(d, x_vals))
    @test mean(abs.(est_lpdf - logpdf.(d, x_vals))) ≈ 0 atol = 5e-1 
end