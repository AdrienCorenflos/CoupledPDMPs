using Distributions: MvNormal, logpdf, rand, Exponential
using LinearAlgebra: norm
using Random: randn, rand

const F = AbstractFloat
const vF = AbstractVector{F}


"""
    coupled_next_refresh!(x₁, x₂, λ, v₁, v₂)

Samples τ from an Exponential distribution with rate λ and a related coupling of multivariate 
standard Gaussians v₁, v₂ such that the probability P(x₁ + τ v₁ = x₂ + τ v₂| τ) is maximal.

I believe this is also maximising P(x₁ + τ v₁ = x₂ + τ v₂) overall, but I need to check.

# Examples
```julia-repl
julia> x₁, x₂ = randn(5), randn(5)
julia> λ = rand()
julia> v₁, v₂ = similar(x₁), similar(x₂)
julia> coupled_next_refresh!(x₁, x₂, λ, v₁, v₂)
0.105, 0.105, false
```
"""
function coupled_next_refresh!(x₁::vF, x₂::vF, λ::F, v₁::vF, v₂::vF)::Tuple{F,F,Bool}
    τ = rand(Exponential()) / λ
    coupled = reflection_maximal!(x₁, x₂, τ, v₁, v₂)

    v₁ -= x₁
    v₂ -= x₂

    v₁ /= τ
    v₂ /= τ

    return τ, τ, coupled
end

"""
    coupled_next_refresh(x₁, x₂, λ)

Pure version of `coupled_next_refresh!`.

See also [`coupled_next_refresh!`](@ref)
"""
function coupled_next_refresh(x₁::vF, x₂::vF, λ::F)
    v₁, v₂ = similar(x₁), similar(x₂)
    τ₁, τ₂, coupled = reflection_maximal!(x₁, x₂, λ, v₁, v₂)
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
function reflection_maximal!(m::vF, μ::vF, σ::F, res₁::vF, res₂::vF)::Bool
    dim = size(m, 1)

    z = (m - μ) ./ sigma
    e = z ./ norm(z)

    n = MvNormal(dim, 1.0)
    ε = rand(n)
    log_u = log(rand())

    temp = ε + z
    ℓₑ = logpdf(n, ε)
    ℓₜ = logpdf(n, temp)

    coupled::Bool = log_u < ℓₜ - ℓₑ
    reflected_ε::vF = coupled ? temp : ε - 2.0 * (ε'e)e

    res₁[:] = m + σ * ε
    res₂[:] = μ + σ * reflected_ε

    return coupled
end

"""
    reflection_maximal(m, μ, σ)

Pure version of `reflection_maximal!`.

See also [`reflection_maximal!`](@ref)
"""
function reflection_maximal(
    m::vF,
    μ::vF,
    σ::F,
) where {F<:AbstractFloat,vF<:AbstractVector{F}}
    x, y = similar(m), similar(μ)
    cond = reflection_maximal!(m, μ, σ, x, y)
    return x, y, cond
end