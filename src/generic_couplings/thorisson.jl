using Distributions
using LinearAlgebra
using Random



"""
    thorisson!(x₁, x₂, d₁, d₂)

Samples from a maximal coupling of two distributions d₁, d₂ using Thorisson's method.

# Examples
```julia-repl
julia> using Distributions, Random
julia> Random.seed!(42)
julia> d₁, d₂ = Normal(0.), Normal(1.)
julia> x₁, x₂ = zeros(1), zeros(1)
julia> x₁, x₂, coupled = thorisson(x₁, x₂, d₁, d₂)
julia> x₁, x₂, coupled
0.7883556016042917, 0.7883556016042917, true
```
"""
function thorisson!(x₁, x₂, d₁::D, d₂::D) where {D<:MultivariateDistribution}
    # Sample from the first distributions.
    rand!(d₁, x₁)
    ℓ₁, ℓ₂ = logpdf(d₁, x₁), logpdf(d₂, x₁)
    u = rand()
    ℓᵤ = log(u)
    if ℓᵤ < ℓ₂ - ℓ₁
        x₂ .= x₁
        return true
    end

    while true
        # Sample uniform
        u = rand()
        ℓᵤ = log(u)
        # Sample from the second distribution.
        rand!(d₂, x₂)

        # Accept condition
        ℓ₁, ℓ₂ = logpdf(d₁, x₂), logpdf(d₂, x₂)
        if ℓᵤ < ℓ₂ - ℓ₁
            return false
        end
    end
end


"""
    thorisson(d₁, d₂)

Samples from a maximal coupling of two distributions d₁, d₂ using Thorisson's method.

# Examples
```julia-repl
julia> using Distributions, Random
julia> Random.seed!(42)
julia> d₁, d₂ = Normal(0.), Normal(1.)
julia> x₁, x₂, coupled = thorisson(d₁, d₂)
0.7883556016042917, 0.7883556016042917, true
```
"""
function thorisson(d₁::D, d₂::D) where {D<:MultivariateDistribution}
    x₁, x₂ = zeros(eltype(d₁), size(d₁)), zeros(eltype(d₂), size(d₂))
    coupled = thorisson!(x₁, x₂, d₁, d₂)
    return x₁, x₂, coupled
end

"""
    thorisson(d₁, d₂)

Samples from a maximal coupling of two distributions d₁, d₂ using Thorisson's method.

# Examples
```julia-repl
julia> using Distributions, Random
julia> Random.seed!(42)
julia> d₁, d₂ = Normal(0.), Normal(1.)
julia> x₁, x₂, coupled = thorisson(d₁, d₂)
0.7883556016042917, 0.7883556016042917, true
```
"""
function thorisson(d₁::D, d₂::D) where {D<:UnivariateDistribution}
    # Sample from the first distributions.
    x₁ = rand(d₁)
    x₂ = copy(x₁)
    ℓ₁, ℓ₂ = logpdf(d₁, x₁), logpdf(d₂, x₁)
    u = rand()
    ℓᵤ = log(u)
    if ℓᵤ < ℓ₂ - ℓ₁
        return x₁, x₂, true
    end

    while true
        # Sample uniform
        u = rand()
        ℓᵤ = log(u)
        # Sample from the second distribution.
        x₂ = rand(d₂)

        # Accept condition
        ℓ₁, ℓ₂ = logpdf(d₁, x₂), logpdf(d₂, x₂)
        if ℓᵤ > ℓ₁ - ℓ₂
            return x₁, x₂, false
        end
    end
end
