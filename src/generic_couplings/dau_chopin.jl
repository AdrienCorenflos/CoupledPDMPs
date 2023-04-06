using Distributions
using LinearAlgebra
using Random


"""
    dau_chopin!(x₁, x₂, d₁, d₂, Γ!)

Samples from a coupling of two distributions d₁, d₂ using Dau and Chopin's method.
This requires a coupling function Γ! that samples from *any* coupling distribution in the first place.
"""
function dau_chopin!(x₁, x₂, d₁::D, d₂::D, Γ!) where {D<:MultivariateDistribution}
    Γ!(x₁, x₂)

    ℓ₁, ℓ₂ = logpdf(d₁, x₁), logpdf(d₂, x₂)
    ℓᵤ, ℓᵥ = log(rand()), log(rand())

    ℓ¹ᵤ = ℓᵤ + ℓ₁
    ℓ²ᵤ = ℓᵤ + ℓ₂

    y = similar(x₁)
    rand!(d₁, y)

    success = 0

    if ℓᵥ < logpdf(y, d₂) - logpdf(y, d₁)
        if ℓ¹ᵤ < logpdf(x₁, d₂)
            success += 1
            x₁ .= y
        end
        if ℓ²ᵤ < logpdf(x₂, d₁)
            success += 1
            x₂ .= y
        end
    end
    return success > 1
end


function dau_chopin(d₁::D, d₂::D, Γ!) where {D<:MultivariateDistribution}
    x₁, x₂ = zeros(eltype(d₁), size(d₁)), zeros(eltype(d₂), size(d₂))
    coupled = dau_chopin!(x₁, x₂, d₁, d₂, Γ!)
    return x₁, x₂, coupled
end


function dau_chopin(d₁::D, d₂::D, Γ) where {D}
    x₁, x₂ = Γ()

    ℓ₁, ℓ₂ = logpdf(d₁, x₁), logpdf(d₂, x₂)
    ℓᵤ, ℓᵥ = log(rand()), log(rand())

    ℓ¹ᵤ = ℓᵤ + ℓ₁
    ℓ²ᵤ = ℓᵤ + ℓ₂

    y = rand(d₁)

    success = 0

    if ℓᵥ < logpdf(d₂, y) - logpdf(d₁, y)
        if ℓ¹ᵤ < logpdf(d₂, x₁)
            success += 1
            x₁ = y
        end
        if ℓ²ᵤ < logpdf(d₁, x₂)
            success += 1
            x₂ = y
        end
    end
    return x₁, x₂, success > 1
end