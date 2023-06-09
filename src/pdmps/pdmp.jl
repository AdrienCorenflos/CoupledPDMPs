include("../poisson/affine.jl")
include("../poisson/homogeneous.jl")
include("../generic_couplings/thorisson.jl")
include("../generic_couplings/exponentials.jl")
include("../coupled_refresh.jl")
using LinearAlgebra: norm

struct coupled_event_info
    """ Event information for next coupled event
    Next event times, bounce indicators, flags for coupled time, position and bounce
    """
    τ₁::Float64
    τ₂::Float64
    bounce₁::Bool
    bounce₂::Bool
    ref_coupled::Bool
    ref_pos_coupled::Bool
    b_coupled::Bool
end

struct coupled_pdmp
    """ Init, Kernel and Coupled Kernel
    Give a function to initialise the sampler and set the kernel and coupled kernel
    """
    init::Function
    kernel::Function
    coupled_kernel::Function
    dynamics::Function
    Δt::Float64
    λᵣ::Float64
    couple_mode::AbstractString
end

struct Kernel
    """ Init, Kernel
    Give a function to initialise the sampler and set the kernel and coupled kernel
    """
    init::Function
    kernel::Function
end


#############################################
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

function dau_chopin(d₁::D, d₂::D, Γ) where {D}
    x₁, x₂, _ = Γ()

    ℓ₁, ℓ₂ = logpdf(d₁, x₁), logpdf(d₂, x₂)
    ℓᵤ, ℓᵥ = log(rand()), log(rand())

    ℓ¹ᵤ = ℓᵤ + ℓ₁
    ℓ²ᵤ = ℓᵤ + ℓ₂

    y = rand(d₁)

    success = 0

    if ℓᵥ < logpdf(d₂, y) - logpdf(d₁, y)
        if ℓ¹ᵤ < logpdf(d₂, x₁)
            success += 1
            x₁ = y
        end
        if ℓ²ᵤ < logpdf(d₁, x₂)
            success += 1
            x₂ = y
        end
    end
    return x₁, x₂, success > 1
end

function modified_lindvall_roger(mx, my, sigmax, sigmay)
    function Γ()
        return lindvall_roger(mx, my, sigmax, sigmay)
    end
    d₁, d₂ = MvNormal(mx, sigmax), MvNormal(my, sigmay)
    return dau_chopin(d₁, d₂, Γ)
end

function coupling_refresh(rate, shift, mode="independent")
    # We want to couple t_1 = t_2 + shift, where both t_1 and t_2 are 
    # exponentially distributed with the same rate.
    # Swap t_1 and t_2 if shift negative
    if shift < 0. 
        t_2, t_1, coupled = coupling(rate, -shift, mode)
    else
        t_1, t_2, coupled = coupling(rate, shift, mode)
    end
    return t_1, t_2, coupled
end
function coupling_bounce(thin_1::AffinePoisson, thin_2::AffinePoisson)
    τb₁, τb₂, coupled = thorisson(thin_1, thin_2)
    return τb₁, τb₂ - thin_2.shift, coupled
end
function bounce(v, grad)
    nrm = norm(grad, 2)
    if nrm ≈ 0
        e = sign.(nrm) 
        norm_e = norm(e)
        if norm_e != 0 
            e ./= norm_e
        else
            e = zeros(length(v))
        end
    else
        e = grad ./ nrm
    end        
    return v - 2 * sum(e .* v) * e
end
#################################################
