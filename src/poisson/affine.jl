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
struct AffinePoisson{T<:AbstractFloat} <: PoissonProcess
    a::T
    b::T
    c::T
    shift::T
    AffinePoisson{T}(a::T, b::T, c::T, shift::T) where {T<:Real} = new{T}(a, b, c, shift)
end

function AffinePoisson(a::T, b::T) where {T<:Real}
    return AffinePoisson{T}(a, b, T(0.0), T(0.0))
end

function AffinePoisson(a::T, b::T, c::T) where {T<:Real}
    return AffinePoisson{T}(a, b, c, T(0.0))
end

function AffinePoisson(a::T, b::T, c::T, shift::T; check_args::Bool = true) where {T<:Real}
    @check_args AffinePoisson -Inf < a < Inf
    @check_args AffinePoisson -Inf < b < Inf
    @check_args AffinePoisson 0.0 <= c < Inf
    @check_args AffinePoisson -Inf <= shift < Inf
    return AffinePoisson{T}(a, b, c, shift)
end

### Parameters
partype(::AffinePoisson{T}) where {T} = T

### Methods
function rand(rng::AbstractRNG, d::AffinePoisson{T}) where {T<:AbstractFloat}
    # See ...
    a, b, c, shift = d.a, d.b, d.c, d.shift
    logᵤ = log(rand(rng, T))

    if isapprox(a, 0., atol=1e-18) # λ = (b)₊ + c
        return shift-logᵤ/(max(0.,b)+c)

    elseif (b < 0.0) & (a > 0.0)
        t₀ = -b/a
        t₁ = -logᵤ / c

        if t₁ < t₀ 
            # λ = c on (0,-b/a)      
            return shift + t₁
        else 
            # λ = (at+b)₊ + c on (-b/a, t) -> λ = (at)₊ + c on (0, t)
            t₂ = (-a * (b + c) + sqrt(a^2*(c^2-2*a*logᵤ+2*a*c*(b/a))))/a^2
            return shift + t₂
        end

    elseif (b > 0.0) & (a < 0.0)
        t₀ = -b/a
        if a*t₀^2/2.0+(b+c)*t₀ <= -logᵤ 
            return shift + (b^2/(2*a) - logᵤ)/c
        else 
            # λ = at+(b+c) on (0, -b/a)
            return shift -(b+c)/a -sqrt((b+c)^2/a^2 - 2*logᵤ/a)
        end

    elseif (a > 0.) & (b > 0.) # λ = at+(b+c) 
        return shift -(b+c)/a + sqrt(((b+c)/a)^2 - 2*logᵤ/a)
        
    else # λ = c
        return shift -logᵤ/c
    end

end


function logpdf(d::AffinePoisson, x::Real)
    
    a, b, c, shift = d.a, d.b, d.c, d.shift
    x = x - shift

    if x < 0.0
        return -Inf
    elseif isapprox(a, 0.0, atol = 1e-18)
        # λ = (b)₊ + c 
        return log(max(0.0, b) + c) - (max(0.0, b) + c) * x

    elseif (b < 0.0) & (a > 0.0) 
        # log Norm constant
        lnc = log( exp(b*c/(a)) + 1.0 - exp(c * b / a) )  

        if x < -b / a
            # λ = c on (0,-b/a)      
            return log(c) - c * x - lnc
        else
            # λ = (at+b) + c on (-b/a, t)
            return log(a*x+b+c)-( -b*c/a+ (b+a*x)*(b + 2*c + a*x)/(2*a) ) - lnc
            #return log(a*x+b+c)-(a * x^2 / 2 + b * x + c * x + b*c/a+ b^2/(2*a)) - lnc
        end

    elseif (b > 0.0) & (a < 0.0) # Issue
        # log Norm constant
        int1 =  erf((b + c)*im / sqrt(2 * -a))*im - erf(c*im / sqrt(2 * abs(a)))*im
        int2 = exp((b+c)^2/(2*a))
        int3 = 2.0*Complex(a)^(3.0/2.0)

        nc = -a *(
            exp(b*(b+2*c)/(2*a))/a - 1/a -sqrt(2*pi)*erf((b+c)/sqrt(Complex(a)*2))*int2*(b+c)/int3+sqrt(2*pi)*erf(c/sqrt(2*Complex(a)))*int2*(b+c)/int3) -
            b*sqrt(pi/(-2*a))*int2*int1-c*sqrt(pi/(-2*a))*int2*int1
        
        lnc = log(real(nc) + exp(c * b / a + b^2/(2*a))) # (-b/a, inf)

        if x < -b / a
            # λ = at+(b+c) on (0, -b/a)
            return log(a*x+b+c) - (a * x^2 / 2 + b * x + c * x) - lnc
        else
            # λ = c on (-b/a, inf)
            return log(c) -c * x +b^2/(2*a) - lnc
        end

    elseif (a > 0.0) & (b > 0.0) # λ = at+(b+c) 
        return log(a * x + b + c) - (a * x^2 / 2 + (b + c) * x)

    else
        # λ = c
        if c <= 0.0
            throw(ArgumentError("AffinePoisson: c must be positive if a = b = 0."))
        end
        return log(c) - c * x
    end
end
rate(d::AffinePoisson, t::Real) = max(d.a * t + d.b, 0.) + d.c