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
    λᵣ::Float64
    Δt::Float64
    ΔM::Int
    accumulate_h::Function
    h::Function
    couple_mode::AbstractString
end

struct Kernel
    """ Init, Kernel
    Give a function to initialise the sampler and set the kernel and coupled kernel
    """
    init::Function
    kernel::Function
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
