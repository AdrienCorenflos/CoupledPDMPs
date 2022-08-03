using Random
using Statistics

@testset "Check that the reflection maximal coupling samples from the right marginals and is maximal." 
begin
    Random.seed!(123456)
    N, d = 10_000, 5
    x₁, x₂ = randn(d), randn(d)
    σ = rand()

    res₁ = zeros(Float64, (d, N))
    res₂ = zeros(Float64, (d, N))
    coupled = zeros(Bool, (N,))
    for i in 1:N
        coupled[i] = coupled_next_refresh!(x₁, x₂, σ, res₁[:, i], res₂[:, i])
    end

    println("coupled:", mean(coupled))
    println("mean₁", mean(res₁, 2))
    println("mean₂", mean(res₂, 2))
end