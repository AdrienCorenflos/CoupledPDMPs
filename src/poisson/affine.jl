include("base.jl")

"""
AffinePoisson(λ)
A distribution describing the next event of an *Affine Poisson process* with affine rate λ(t) = (a*t + b)₊ .
```math
PP(t; \\lambda) = \\lambda(t) \\exp(-\\int_{0}^t\\lambda(s)\\mathrm{d}s), \\quad t \\in [0, \\infty]
```

```julia
AffinePoisson(a, b)  # Affine Poisson process with λ(t) = c + (a*t + b)₊
```
"""
struct AffinePoisson{T<:Real} <: PoissonProcess
    a::T
    b::T
    c::T
    AffinePoisson{T}(a::T, b::T, c::T) where {T<:Real} = new{T}(a, b, c)
end

function AffinePoisson(a::T, b::T) where {T<:Real}
    return AffinePoisson{T}(a, b, T(0.0))
end

function AffinePoisson(a::T, b::T, c::T; check_args::Bool = true) where {T<:Real}
    @check_args AffinePoisson -Inf < a < Inf
    @check_args AffinePoisson -Inf < b < Inf
    @check_args AffinePoisson 0.0 <= c < Inf
    return AffinePoisson{T}(a, b, c)
end
### Parameters
partype(::AffinePoisson{T}) where {T} = T

### Methods
function rand(rng::AbstractRNG, d::AffinePoisson{T}) where {T}
    # See ...
    a, b, c = d.a, d.b, d.c
    logᵤ = log(rand(rng, T))
    if isapprox(b, 0.0, atol = 1e-10)
        if a > 0
            return -logᵤ / (a + c)
        else
            return -logᵤ / c
        end
    elseif b > 0
        if a < 0
            if -a * c / b + logᵤ < 0.0
                return sqrt(-2 * b * logᵤ + c^2 + 2 * a * c) / b - (a + c) / b
            else
                return -logᵤ / c
            end
        else
            return sqrt(-logᵤ * 2.0 * b + (a + c)^2) / b - (a + c) / b
        end
    else
        if a <= 0.0
            return -logᵤ / c
        elseif -a * c / b - a^2 / (2 * b) + logᵤ > 0.0
            return +sqrt((a + c)^2 - 2.0 * logᵤ * b) / b - (a + c) / b
        else
            return (-logᵤ + a^2 / (2 * b)) / c
        end
    end
end


function logpdf(d::AffinePoisson, x::Real)
    error("Not implemented")
    a, b, c = d.a, d.b, d.c
    if x < 0.0
        return -Inf
    elseif isapprox(b, 0.0, atol = 1e-10)
        return log(a + c) - (a + c) * x
    elseif b > 0
        if a < 0
            if x < -a * c / b
                return -Inf
            else
                return log(a + c) - (a + c) * x
            end
        else
            return log(a + c) - (a + c) * x
        end
    else
        if a <= 0.0
            return log(c) - c * x
        elseif x < -a * c / b
            return -Inf
        else
            return log(c) - c * x
        end
    end
end
    
rate(d::AffinePoisson, t::Real) = maximum(d.a * t + d.b, 0) + d.c