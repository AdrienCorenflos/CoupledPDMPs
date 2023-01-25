using Distributions
using Random


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
function thorisson(d₁, d₂)
    # Sample from the first distributions.
    x₁ = rand(d₁)
    ℓ₁, ℓ₂ = logpdf(d₁, x₁), logpdf(d₂, x₁)
    u = rand()
    ℓᵤ = log(u)
    if ℓᵤ < ℓ₂ - ℓ₁
        x₂ = x₁
        return x₁, x₂, true
    end
    U = Uniform()
    while true
        # Sample uniform
        u = rand()
        ℓᵤ = log(u)
        # Sample from the second distribution.
        x₂ = rand(d₂)
        
        # Accept condition
        ℓ₁, ℓ₂ = logpdf(d₁, x₂), logpdf(d₂, x₂)
        if ℓᵤ < ℓ₁ - ℓ₂
            return x₁, x₂, false
        end
    end
end
