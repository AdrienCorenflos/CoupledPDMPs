include("base.jl")
using SpecialFunctions

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

    if isapprox(a, 0., atol=1e-10) # λ = (b)₊ + c
        return -logᵤ/(max(0.,b)+c)

    elseif (b < 0.) & (a > 0.)    
        t₀ = -logᵤ/c
        if t₀ < -b/a
            # λ = c on (0,-b/a)      
            return t₀
        else 
            # λ = (at+b)₊ + c on (-b/a, t) -> λ = (at)₊ + c on (0, t)
            logᵤ = logᵤ + b*c/a 
            return -b/a - c/a + sqrt((c/a)^2 - 2*logᵤ/a)
        end

    elseif (b > 0.) & (a < 0.)

        if -b^2/a + b^2/(2*a) >= -logᵤ + c*(b/a)
            # λ = at+(b+c) on (0, -b/a)
            return -(b+c)/a -sqrt(((b+c)/a)^2 - 2*logᵤ/a)
        else 
            logᵤ = logᵤ - c*(b/a)
            return -b/a + -logᵤ/c
        end

    elseif (a > 0.) & (b > 0.) # λ = at+(b+c) 
        return -(b+c)/a + sqrt(((b+c)/a)^2 - 2*logᵤ/a)
        
    else # λ = c
        return -logᵤ/c
    end

end


function logpdf(d::AffinePoisson, x::Real)
    
    a, b, c = d.a, d.b, d.c
    if x < 0.0
        return -Inf
    elseif isapprox(a, 0.0, atol = 1e-10)
        # λ = (b)₊ + c 
        return log(max(0.,b) + c) - (max(0.,b) + c) * x

    elseif (b < 0.) & (a > 0.)    
        # log Norm constant
        lnc = 0.5*log(pi/(2*a)) + (b+c)^2/(2*a) +log(1-erf(c/sqrt(2*a)))
        lnc = log(exp(lnc) +1/c-exp(c*b/a)/c )  # (0, -b/a)

        if x < -b/a
            # λ = c on (0,-b/a)      
            return -c*x - lnc
        else 
            # λ = (at+b) + c on (-b/a, t)
            return -(a*x^2/2 +b*x + c*x) - lnc
        end

    elseif (b > 0.) & (a < 0.)
        # log Norm constant
        lnc = sqrt(pi/abs(2*a))*exp((b+c)^2/(2*a))*(erfi((b+c)/sqrt(2*abs(a))) -erfi(c/sqrt(2*abs(a))))# (0,-b/a)
        lnc = log(lnc + exp(c*b/a)/c ) # (-b/a, inf)

        if x < -b/a
            # λ = at+(b+c) on (0, -b/a)
            return -(a*x^2/2 +b*x + c*x) - lnc
        else 
            # λ = c on (-b/a, inf)
            return -c*x - lnc
        end

    elseif (a > 0.) & (b > 0.) # λ = at+(b+c) 
        lnc = sqrt(pi/(2*a))*exp((b+c)^2/(2*a))*erfc((b+c)/sqrt(2*abs(a)))
        return -(a*x^2/2 +b*x + c*x) - lnc

    else
        # λ = c
        return log(c) - c*x 
    end
end
    
rate(d::AffinePoisson, t::Real) = maximum(d.a * t + d.b, 0) + d.c