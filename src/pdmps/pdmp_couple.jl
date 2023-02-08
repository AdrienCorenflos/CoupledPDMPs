include("pdmp.jl")
include("../poisson/homogeneous.jl")
include("../generic_couplings/thorisson.jl")
using CoupledPDMPs
abstract type Coupledpdmp end

mutable struct CoupledBPS <: Coupledpdmp
    sampler::BPS
    t1::Float64
    x1::Vector
    v1::Vector
    t2::Float64
    x2::Vector
    v2::Vector
    Δt::Float64
    is_time_coupled::Bool
    is_position_coupled::Bool
    is_velocity_coupled::Bool
end
"""
    CoupledBPS(BPS, x, v)

Implementation of a coupled BPS sampler. Given a gradient function ∇U, and H
an upper bound on the Hessian run one event forward using the BPS sampler.

# Examples
```julia-repl
julia> function ∇U(x::Vector)
julia> x
julia> end
julia> H = [1 0; 0  1]
julia> test = BPS(∇U, H, 1.)
julia> testC = CoupledBPS(test, randn(2), randn(2))
julia> update(testC)
(false, false, false)
```
"""


function CoupledBPS(sampler::BPS, x::Vector, v::Vector)

    t1, x1, v1 = update(sampler, 0., x, v)
    t2, x2, v2 = 0., randn(size(x1)), randn(size(v1))
    Δt = t1

    return CoupledBPS(sampler, t1, x1, v1, t2, x2, v2, Δt, false, false, false)
end

function current_state(coupling::Coupledpdmp)

    return coupling.t1, coupling.x1, coupling.v1, coupling.t2, coupling.x2, coupling.v2
end


function update(coupling::Coupledpdmp)
    
    t1, x1, v1, t2, x2, v2 = coupling.t1, coupling.x1, coupling.v1, coupling.t2, coupling.x2, coupling.v2

    if(coupling.is_time_coupled && coupling.is_position_coupled && coupling.is_velocity_coupled)
        t1, x1, v1 = update(coupling.sampler, t1, x1, v1)
        t2, x2, v2 = t1 - coupling.Δt, x1, v1
    else
        sampler = coupling.sampler
        
        refresh1, refresh2 = HomogeneousPoisson(sampler.ρ), HomogeneousPoisson(sampler.ρ, t2+coupling.Δt-t1)
        
        τᵣ¹, τᵣ², coupling.is_time_coupled = thorisson(refresh1, refresh2)
        τᵣ² = τᵣ² - (t2+coupling.Δt-t1)
        τₑ¹, τₑ²  = bounce_thin(sampler, x1, v1), bounce_thin(sampler, x1, v1)
        
        τᵣ, τₑ = [τᵣ¹, τᵣ²], [τₑ¹, τₑ²]
        τ = min.(τₑ, τᵣ)

        t1, t2 = t1+τ[1], t2+τ[2]
        x1, x2 = x1+v1*τ[1], x2+v2*τ[2]

        if( all(τᵣ .< τₑ) )
            if(coupling.is_time_coupled)
                current_rng = copy(Random.default_rng())
                coupled_next_refresh!(x1, x2, sampler.ρ, v1, v2)
                copy!(Random.default_rng(), current_rng)
            else 
                reflection_maximal!(fill(0.,size(v1)), fill(0.,size(v1)), 1., v1, v2)
            end
        else
            print("event")
            coupling.is_time_coupled = false
            v1 = event_update(sampler, x1, v1, τᵣ[1] < τₑ[1])
            v2 = event_update(sampler, x2, v2, τᵣ[2] < τₑ[2])
        end
            
        coupling.is_position_coupled = max(norm(x1 - x2)) < 1e-10
        coupling.is_velocity_coupled = max(norm(v1 - v2)) < 1e-10
    end
    coupling.t1, coupling.x1, coupling.v1, coupling.t2, coupling.x2, coupling.v2 = t1, x1, v1, t2, x2, v2

    (coupling.is_time_coupled, coupling.is_position_coupled, coupling.is_velocity_coupled)
end

