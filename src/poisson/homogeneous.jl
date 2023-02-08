include("base.jl")
 
"""
    HomogeneousPoisson(λ)
A distribution describing the next event of a *Homogeneous Poisson process* with constant rate λ.
```math
PP(t; \\lambda) = \\lambda \\exp(-\\lambda t), \\quad t \\in [0, \\infty)
```

```julia
HomogeneousPoisson(λ)  # Homogeneous Poisson process λ
```
"""
struct HomogeneousPoisson{T<:Real} <: PoissonProcess
    exp::Exponential{T}
    shift::T
    HomogeneousPoisson{T}(λ::T, shift::T) where {T<:Real} = new{T}(Exponential(1 / λ), shift)
end

function HomogeneousPoisson(λ::T, shift::T; check_args::Bool=true) where {T <: Real}
    # @check_args HomogeneousPoisson 0.0 < λ < Inf
    return HomogeneousPoisson{T}(λ, shift)
end
function HomogeneousPoisson(λ::T; check_args::Bool=true) where {T <: Real}
    # @check_args HomogeneousPoisson 0.0 < λ < Inf
    return HomogeneousPoisson{T}(λ, 0.0)
end
### Parameters
partype(::HomogeneousPoisson{T}) where {T} = T

### Methods
rand(rng::AbstractRNG, d::HomogeneousPoisson) = rand(rng, d.exp)+d.shift
logpdf(d::HomogeneousPoisson, x::Real) = logpdf(d.exp, x-d.shift)
cdf(d::HomogeneousPoisson, x::Real) = cdf(d.exp, x-d.shift)
quantile(d::HomogeneousPoisson, p::Real) = quantile(d.exp, p) + d.shift
minimum(d::HomogeneousPoisson) = d.shift
maximum(d::HomogeneousPoisson) = Inf
insupport(d::HomogeneousPoisson, x::Real) = insupport(d.exp, x-d.shift)