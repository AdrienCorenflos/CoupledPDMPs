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
    HomogeneousPoisson{T}(λ::T) where {T<:Real} = new{T}(Exponential(1 / λ))
end

function HomogeneousPoisson(λ::T; check_args::Bool=true) where {T <: Real}
    @check_args HomogeneousPoisson 0.0 < λ < Inf
    return HomogeneousPoisson{T}(λ)
end
### Parameters
partype(::HomogeneousPoisson{T}) where {T} = T

### Methods
rand(rng::AbstractRNG, d::HomogeneousPoisson) = rand(rng, d.exp)
logpdf(d::HomogeneousPoisson, x::Real) = logpdf(d.exp, x)
cdf(d::HomogeneousPoisson, x::Real) = cdf(d.exp, x)
quantile(d::HomogeneousPoisson, p::Real) = quantile(d.exp, p)
minimum(d::HomogeneousPoisson) = 0.0
maximum(d::HomogeneousPoisson) = Inf
insupport(d::HomogeneousPoisson, x::Real) = insupport(d.exp, x)