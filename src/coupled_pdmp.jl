using Distributions, StatsBase, LinearAlgebra, Random, Plots
include("exp_couple.jl")
include("gaussian_couple.jl")
include("couple_bps.jl")

d = 2
t1, x1, t2, x2, dt = bps_coupled(randn(2),randn(2),randn(2),randn(2),1,1, 1000)
plot(x1[:,1], x1[:,2])
plot(x2[:,1], x2[:,2])

plot(t1, x1[:,1])
plot!(t2, x2[:,1])