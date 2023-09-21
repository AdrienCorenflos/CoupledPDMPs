using Distributions: MvNormal, logpdf, rand, Exponential
using LinearAlgebra: norm
using Random: randn, rand

function coupling_refresh(rate, shift, mode="independent")
    # Coupling refreshment times 
    if shift < 0. 
        t_2, t_1, coupled = coupling(rate, -shift, mode)
    else
        t_1, t_2, coupled = coupling(rate, shift, mode)
    end
    return t_1, t_2, coupled
end

function coupling_bounce(thin_1::AffinePoisson, thin_2::AffinePoisson)
    # Coupling bounce event (requires Affine coupling)
    τb₁, τb₂, coupled = thorisson(thin_1, thin_2)
    return τb₁, τb₂ - thin_2.shift, coupled
end

"""
    coupled_next_refresh!(x₁, x₂, λ, v₁, v₂)

Samples τ from an Exponential distribution with rate λ and a related coupling of multivariate 
standard Gaussians v₁, v₂ such that the probability P(x₁ + τ v₁ = x₂ + τ v₂| τ) is maximal.

# Examples
```julia-repl
julia> x₁, x₂ = randn(5), randn(5)
julia> λ = rand()
julia> v₁, v₂ = similar(x₁), similar(x₂)
julia> coupled_next_refresh!(x₁, x₂, λ, v₁, v₂)
0.105, 0.105, false
```
"""
function coupled_next_refresh!(
    x₁::AbstractVector{F},
    x₂::AbstractVector{F},
    λ::F,
    v₁::AbstractVector{F},
    v₂::AbstractVector{F},
) where {F<:AbstractFloat}

    τ = rand(Exponential()) / λ
    coupled = reflection_maximal!(x₁, x₂, τ, v₁, v₂)

    v₁ .-= x₁
    v₂ .-= x₂

    v₁ /= τ
    v₂ /= τ

    return τ, τ, coupled
end


"""
    coupled_next_refresh(x₁, x₂, λ)

Pure version of `coupled_next_refresh!`.

See also [`coupled_next_refresh!`](@ref)
"""
function coupled_next_refresh(
    x₁::AbstractVector{F},
    x₂::AbstractVector{F},
    λ::F,
) where {F<:AbstractFloat}
    v₁, v₂ = similar(x₁), similar(x₂)
    τ₁, τ₂, coupled = coupled_next_refresh!(x₁, x₂, λ, v₁, v₂)
    return v₁, v₂, τ₁, τ₂, coupled
end

# Reflection maximal coupling for MVNs with isotropic noise.


"""
    reflection_maximal!(m, μ, σ, res₁, res₂)

Maximal coupling of multivariate standard Gaussians m, μ with isotropic covariance σ⋅Id.

# Examples
```julia-repl
julia> m, μ = randn(5), randn(5)
julia> σ = rand()
julia> res₁, res₂ = similar(m), similar(μ)
julia> reflection_maximal!(m, μ, σ, res₁, res₂)
true
```
"""
function reflection_maximal!(
    m::AbstractVector{F},
    μ::AbstractVector{F},
    σ::F,
    res₁::AbstractVector{F},
    res₂::AbstractVector{F},
) where {F<:AbstractFloat}
    dim = size(m, 1)

    z = (m - μ) / σ
    e = z / norm(z)

    n = MvNormal(dim, 1.0)
    ε = rand(n)
    log_u = log(rand())

    z .+= ε
    ℓₑ = logpdf(n, ε)
    ℓₜ = logpdf(n, z)

    coupled::Bool = log_u < (ℓₜ - ℓₑ)
    # Reuse z for memory efficiency
    z .= coupled ? z : ε .- 2(ε'e)e

    @. res₁ = m + σ * ε
    @. res₂ = μ + σ * z

    return coupled
end

"""
    reflection_maximal(m, μ, σ)

Pure version of `reflection_maximal!`.

See also [`reflection_maximal!`](@ref)
"""
function reflection_maximal(
    m::AbstractVector{F},
    μ::AbstractVector{F},
    σ::F,
) where {F<:AbstractFloat}
    x, y = similar(m), similar(μ)
    cond = reflection_maximal!(m, μ, σ, x, y)
    return x, y, cond
end

"""
    reflection_maximal(m, μ, Q_x, Q_x_inv, Q_y, res₁, res₂)
Maximal coupling of multivariate Gaussians m, μ with Σₓ = QₓQₓᵀ Σ_y = Q_yQ_yᵀ

```
"""
function reflection_maximal(
    mx::AbstractVector{F}, 
    my::AbstractVector{F}, 
    Q_x::AbstractMatrix{F}, 
    Q_x_inv::AbstractMatrix{F}, 
    Q_y::AbstractMatrix{F}
    ) where {F<:AbstractFloat}

    # Get dimension
    d = size(mx)[1]

    # Get scaled difference
    z = Q_x_inv*(mx - my)  # Q^{-1/2}(mx - my)
    if mx ≈ my
        e = sign.(z) 
        norm_e = norm(e)
        if norm_e != 0 
            e ./= norm_e
        else
            e = 1/sqrt(d).* ones(d)
        end
    else
        e = z ./ norm(z)
    end

    n = MvNormal(d, 1.0)
    ε_x = rand(n)
    log_u = log(rand())

    z .+= ε_x
    ℓₑ = logpdf(n, ε_x)
    ℓₜ = logpdf(n, z)

    coupled::Bool = log_u < (ℓₜ - ℓₑ)
    
    # Reuse z for memory efficiency
    z .= coupled ? z : ε_x .- 2(ε_x'e)e

    # Sample
    x = mx .+ Q_x * ε_x
    y = my .+ Q_y * z

    return x, y, coupled
end


function lindvall_roger(mx, my, Q_x, Q_x_inv, Q_y)
    # Get dimension
    d = size(mx)[1]

    # Get scaled difference
    z = Q_x_inv*(mx - my)  # Q^{-1/2}(mx - my)
    if mx ≈ my
        e = sign.(z) 
        norm_e = norm(e)
        if norm_e != 0 
            e ./= norm_e
        else
            e = 1/sqrt(d).* ones(d)
        end
    else
        e = z ./ norm(z)
    end

    # Get x noise
    eps_x = randn(d)

    # Get y noise
    eps_y = eps_x - 2 * dot(e, eps_x) * e  # eps_y = eps_x - 2<e, eps_x>e, the reflection. 
    
    # Sample
    x = mx .+ Q_x * eps_y
    y = my .+ Q_y * eps_x

    return x, y, false
end

function lindvall_roger(mx, my, sigmax, sigmay)
    # Get dimension
    d = size(mx)[1]

    # Get scaled difference
    z = (mx - my) ./ sigmay  # Q^{-1/2}(mx - my)
    if mx ≈ my
        e = sign.(z) 
        norm_e = norm(e)
        if norm_e != 0 
            e ./= norm_e
        else
            e = 1/sqrt(d).* ones(d)
        end
    else
        e = z ./ norm(z)
    end

    # Get x noise
    eps_x = randn(d)

    # Get y noise
    eps_y = eps_x - 2 * dot(e, eps_x) * e  # eps_y = eps_x - 2<e, eps_x>e, the reflection. 
    
    # Sample
    x = mx .+ sigmax .* eps_x
    y = my .+ sigmay .* eps_y

    return x, y, false
end
