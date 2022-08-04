using Distributions: MvNormal, logpdf, rand, Exponential
using LinearAlgebra: norm
using Random: randn, rand


"""
    coupled_next_refresh!(x₁, x₂, λ, v₁, v₂)

Samples τ from an Exponential distribution with rate λ and a related coupling of multivariate 
standard Gaussians v₁, v₂ such that the probability P(x₁ + τ v₁ = x₂ + τ v₂| τ) is maximal.

# Examples
```julia-repl
julia> x₁, x₂ = randn(5), randn(5)
julia> λ = rand()
julia> v₁, v₂ = similar(x₁), similar(x₂)
julia> coupled_next_refresh!(x₁, x₂, λ, v₁, v₂)
0.105, 0.105, false
```
"""
function coupled_next_refresh!(
    x₁::AbstractVector{F},
    x₂::AbstractVector{F},
    λ::F,
    v₁::AbstractVector{F},
    v₂::AbstractVector{F},
) where {F<:Real}

    τ = rand(Exponential()) / λ
    coupled = reflection_maximal!(x₁, x₂, τ, v₁, v₂)

    v₁ .-= x₁
    v₂ .-= x₂

    v₁ /= τ
    v₂ /= τ

    return τ, τ, coupled
end


"""
    coupled_next_refresh(x₁, x₂, λ)

Pure version of `coupled_next_refresh!`.

See also [`coupled_next_refresh!`](@ref)
"""
function coupled_next_refresh(
    x₁::AbstractVector{F},
    x₂::AbstractVector{F},
    λ::F,
) where {F<:Real}
    v₁, v₂ = similar(x₁), similar(x₂)
    τ₁, τ₂, coupled = coupled_next_refresh!(x₁, x₂, λ, v₁, v₂)
    return v₁, v₂, τ₁, τ₂, coupled
end

# Reflection maximal coupling for MVNs with isotropic noise.


"""
    reflection_maximal!(m, μ, σ, res₁, res₂)

Maximal coupling of multivariate standard Gaussians m, μ with isotropic covariance σ⋅Id.

# Examples
```julia-repl
julia> m, μ = randn(5), randn(5)
julia> σ = rand()
julia> res₁, res₂ = similar(m), similar(μ)
julia> reflection_maximal!(m, μ, σ, res₁, res₂)
true
```
"""
function reflection_maximal!(
    m::AbstractVector{F},
    μ::AbstractVector{F},
    σ::F,
    res₁::AbstractVector{F},
    res₂::AbstractVector{F},
) where {F<:Real}
    dim = size(m, 1)

    z = (m - μ) / σ
    e = z / norm(z)

    n = MvNormal(dim, 1.0)
    ε = rand(n)
    log_u = log(rand())

    z .+= ε
    ℓₑ = logpdf(n, ε)
    ℓₜ = logpdf(n, z)

    coupled::Bool = log_u < (ℓₜ - ℓₑ)
    # Reuse z for memory efficiency
    z .= coupled ? z : ε .- 2 (ε'e) e

    @. res₁ = m + σ * ε
    @. res₂ = μ + σ * z

    return coupled
end


"""
    reflection_maximal(m, μ, σ)

Pure version of `reflection_maximal!`.

See also [`reflection_maximal!`](@ref)
"""
function reflection_maximal(
    m::AbstractVector{F},
    μ::AbstractVector{F},
    σ::F,
) where {F<:Real}
    x, y = similar(m), similar(μ)
    cond = reflection_maximal!(m, μ, σ, x, y)
    return x, y, cond
end
