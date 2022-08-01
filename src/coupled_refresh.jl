using Distributions: MvNormal, logpdf, rand, Exponential
using LinearAlgebra: norm
using Random: randn, rand

const F = AbstractFloat
const vF = AbstractVector{F}

function coupled_next_refresh!(x₁::vF, x₂::vF, λ::F, v₁::vF, v₂::vF)::Tuple{F, F,Bool}
    τ = rand(Exponential()) / λ
    coupled = reflection_maximal!(x₁, x₂, τ, v₁, v₂)

    v₁ -= x₁
    v₂ -= x₂

    v₁ /= τ
    v₂ /= τ

    return τ, τ, coupled
end


function coupled_next_refresh(x₁::vF, x₂::vF, λ::F)
    v₁, v₂ = similar(x₁), similar(x₂)
    τ₁, τ₂, coupled = reflection_maximal!(x₁, x₂, λ, v₁, v₂)
    return v₁, v₂, τ₁, τ₂, coupled
end

# Reflection maximal coupling for MVNs with isotropic noise.

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

function reflection_maximal(
    m::vF,
    μ::vF,
    σ::F,
) where {F<:AbstractFloat,vF<:AbstractVector{F}}
    x, y = similar(m), similar(μ)
    cond = reflection_maximal!(m, μ, σ, x, y)
    return x, y, cond
end