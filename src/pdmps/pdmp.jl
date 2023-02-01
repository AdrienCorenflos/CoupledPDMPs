include("temp_poisson.jl")
abstract type pdmp end

"""
    BPS(∇U, H, \rho)

Implementation of a BPS sampler. Given a gradient function ∇U, and H
an upper bound on the Hessian run one event forward using the BPS sampler.

# Examples
```julia-repl
julia> function ∇U(x::Vector)
julia> x
julia> end
julia> H = [1 0; 0  1]
julia> test = BPS(∇U, H, 1.)
julia> update(test, 0., randn(2), randn(2))
(0.5574, [0.3020, -0.8715], [1.0557, -1.1476])
```
"""

struct BPS <: pdmp
    ∇U::Function
    H::Matrix
    ρ::Float64
end

function event_update(sampler::pdmp, x, v, refresh)
    if( !refresh )
        g = sampler.∇U(x)
        g = g / norm(g, 2)
        v -=  2 * sum(g .* v) * g
    else
        v = randn(length(v))
    end
end


function bounce_thin(sampler::pdmp, x, v, grad)
    event = false
    a = v'*grad
    b = v'*sampler.H*v
    t = 0.

    while(!event)
        τ = lin_bound(a, b)

        t += τ
        x += v*τ
        a += b*τ

        grad = sampler.∇U(x)
        event_rate = v'*grad

        if( rand() < event_rate / a)
            x -= v*t
            event = true
        end
        a = event_rate
    end
    return(t)
end


function update(sampler::pdmp, t::Float64, x::Vector, v::Vector)
    
    ∇U, ρ = sampler.∇U, sampler.ρ

    grad = ∇U(x)

    τₑ = bounce_thin(sampler, x, v, grad)
    τᵣ = -log(rand())/ρ

    τ = min(τₑ, τᵣ)

    t += τ
    x += v*τ

    v = event_update(sampler, x, v, τᵣ < τₑ)
    
    return (t, x, v)
end

