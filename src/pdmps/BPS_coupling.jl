include("../generic_couplings/thorisson.jl")
include("../coupled_refresh.jl")
include("pdmp.jl")
"""
State of the Bouncy Particle Sampler

Additional information regarding the thinning procedure is stored for efficiency.

# Fields
- `skeleton::Skeleton`: The skeleton of the PDMP
- `thinning::AffinePoisson`: The proposal for thinning 
"""

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

function lindvall_roger(mx, my, sigmax, sigmay)
    # Get dimension
    d = size(mx)[1]

    # Get scaled difference
    z = (mx - my) ./ sigmay  # Q^{-1/2}(mx - my)
    if(mx ≈ my)
        e = sign.(z) 
        norm_e = norm(e)
        if(norm_e != 0)
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
    x = mx .+ sigmax .* eps_y
    y = my .+ sigmay .* eps_x

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
    d₁, d₂ = MvNormal(mx, sigmax.*I(size(mx)[1])), MvNormal(my, sigmay.*I(size(my)[1]))
    return dau_chopin(d₁, d₂, Γ)
end

function coupling_refresh(rate, shift, mode="independent")
    # We want to couple t_1 = t_2 + shift, where both t_1 and t_2 are 
    # exponentially distributed with the same rate.
    # We assume the shift is positive
    # If not flip the sign of shift and couple t_2 = t_1 + shift (will swap order)
    swap = false
    if(shift < 0. )
        shift = -shift
        swap = true
    end

    # Because the shift is positive and the rate is the same, we know that (in terms of densities)
    # p(t_2 + shift = t) = 0 if t <= shift, 
    # and p(t_1 = t) < p(t_2 + shift = t) = p(t_2 = t - shift) for t > shift.
    # As a consequence, we know what min(p(t_1 = t), p(t_2 + shift = t)) is everywhere and we can integrate it.

    # With probability mixture_weight, the two are coupled.
    mixture_weight = exp(-rate * shift)
    u = rand()

    coupled = u < mixture_weight
    if u < mixture_weight
        log_v = log(rand())
        t_1 = shift - log_v / rate
        t_2 = t_1
    else
        if mode == "independent"
            v = rand()
            w = rand()
        elseif mode == "crn"
            v = rand()
            w = 0. + v  # to avoid reference and force copy
        elseif mode == "antithetic"
            v = rand()
            w = 1. - v
        else
            throw(DomainError())
        end

        v *= (1 - mixture_weight)
        v = 1 - v

        # This should perhaps be done in logspace
        w *= 1 - mixture_weight
        w /= (exp(rate * shift) - 1)
        w = exp(-rate * shift) - w

        log_v = log(v)
        log_w = log(w)
        t_1 = -log_v / rate
        t_2 = -log_w / rate
    end

    if(swap)
        return t_2 - shift, t_1, coupled
    else
        return t_1, t_2 - shift, coupled
    end
end


function coupling_bounce(thin_1::AffinePoisson, thin_2::AffinePoisson)
    τb₁, τb₂, coupled = thorisson(thin_1, thin_2)
    return τb₁, τb₂ - thin_2.shift, coupled
end


function bounce!(v, grad)
    nrm = norm(grad, 2)
    grad = grad / nrm
    v[:] = v - 2 * sum(grad .* v) * grad
end

function get_thin(v::Vector, grad::Vector, H::Matrix, shift::Float64)
        
    # Set the thinning bound
    a = v'*H*v
    b = v'*grad

    return AffinePoisson(a, b, 0.0, shift)
end

"""
    BPS_coupling(∇U::Function, H::Matrix, h::Function, Δt::Float64, λᵣ::Float64)

TBW
"""
function BPS_coupling(∇U::Function, H::Matrix, Δt::Float64, λᵣ::Float64)

    function init_coupling(x₁::Vector, x₂::Vector)

        t₁ = 0.0; t₂ = 0.0
        
        τr₁, τr₂, ref_coupled = coupling_refresh(λᵣ, Δt + t₂ - t₁)
        v₁, v₂, ref_pos_coupled = modified_lindvall_roger(x₁, x₂, τr₁, τr₂) # Try and couple velocities
        v₁ = (v₁- x₁) ./ τr₁; v₂ = (v₂ - x₂) ./ τr₂; 

        thin_1 = get_thin(v₁, ∇U(x₁), H,  0.0)
        thin_2 = get_thin(v₂, ∇U(x₂), H, Δt + t₂ - t₁)

        # Compute next times
        τb₁, τb₂, b_coupled = coupling_bounce(thin_1, thin_2)
        
        τ₁ = min(τr₁, τb₁)
        τ₂ = min(τr₂, τb₂)

        is_bounce₁ = τb₁ < τr₁
        is_bounce₂ = τb₂ < τr₂

        next_event_info = coupled_event_info(τ₁, τ₂, is_bounce₁, is_bounce₂, ref_coupled, ref_pos_coupled, b_coupled)

        state_1 = (t₁, x₁, v₁)
        state_2 = (t₂, x₂, v₂)
        
        return state_1, state_2, next_event_info, false, false
    end

    function kernel_event(state_1, state_2, next_event_info::coupled_event_info, coupled_next, coupled)
        t₁, x₁, v₁ = state_1
        t₂, x₂, v₂ = state_2

        if(next_event_info.bounce₁ & next_event_info.bounce₂)
            #println("1")
            coupled_t = next_event_info.b_coupled
            coupled_x = false; coupled_v = false
            ref_pos_coupled = false

            t₁ += next_event_info.τ₁
            t₂ += next_event_info.τ₂

            x₁ += next_event_info.τ₁*v₁
            x₂ += next_event_info.τ₂*v₂

            grad_1, grad_2 = ∇U(x₁), ∇U(x₂)
            bounce!(v₁, grad_1)
            bounce!(v₂, grad_2)
            
            # Compute next event times. We can do this after the event given they are only bounces
            thin_1 = get_thin(v₁, grad_1, H,  0.0)
            thin_2 = get_thin(v₂, grad_2, H, Δt + t₂ - t₁)
            
            τb₁, τb₂, b_coupled = coupling_bounce(thin_1, thin_2)
            τr₁, τr₂, ref_coupled = coupling_refresh(λᵣ, Δt + t₂ - t₁)

            is_bounce₁ = τb₁ < τr₁
            is_bounce₂ = τb₂ < τr₂

        elseif(next_event_info.bounce₁)
            #println("2")
            coupled_t = false; coupled_x = false; coupled_v = false
            ref_pos_coupled = false

            t₁ += next_event_info.τ₁
            t₂ += next_event_info.τ₂

            x₁ += next_event_info.τ₁*v₁
            x₂ += next_event_info.τ₂*v₂

            grad_1, grad_2 = ∇U(x₁), ∇U(x₂) ## Can improve efficiency
            bounce!(v₁, grad_1)
            randn!(v₂)

            # Compute next event times. We can do this after the event given they are only bounces
            thin_1 = get_thin(v₁, grad_1, H,  0.0)
            thin_2 = get_thin(v₂, grad_2, H, Δt + t₂ - t₁)
            
            τb₁, τb₂, b_coupled = coupling_bounce(thin_1, thin_2)
            τr₁, τr₂, ref_coupled = coupling_refresh(λᵣ, Δt + t₂ - t₁)

            is_bounce₁ = τb₁ < τr₁
            is_bounce₂ = τb₂ < τr₂

        elseif(next_event_info.bounce₂)
            #println("3")
            coupled_t = false; coupled_x = false; coupled_v = false
            ref_pos_coupled = false

            t₁ += next_event_info.τ₁
            t₂ += next_event_info.τ₂

            x₁ += next_event_info.τ₁*v₁
            x₂ += next_event_info.τ₂*v₂

            grad_1, grad_2 = ∇U(x₁), ∇U(x₂) ## Can improve efficiency
            bounce!(v₂, grad_2)
            randn!(v₁)

            # Compute next event times. We can do this after the event given they are only bounces
            thin_1 = get_thin(v₁, grad_1, H,  0.0)
            thin_2 = get_thin(v₂, grad_2, H, Δt + t₂ - t₁)
            
            τb₁, τb₂, b_coupled = coupling_bounce(thin_1, thin_2)
            τr₁, τr₂, ref_coupled = coupling_refresh(λᵣ, Δt + t₂ - t₁)
            

            is_bounce₁ = τb₁ < τr₁
            is_bounce₂ = τb₂ < τr₂

        else
            #println("4")
            coupled_t = next_event_info.ref_coupled 
            coupled_x = next_event_info.ref_pos_coupled

            t₁ += next_event_info.τ₁
            t₂ += next_event_info.τ₂

            x₁ += next_event_info.τ₁*v₁
            x₂ += next_event_info.τ₂*v₂

            # Update the refreshment PRIOR to the velocity!
            #println(Δt + t₂ - t₁)
            τr₁, τr₂, ref_coupled = coupling_refresh(λᵣ, Δt + t₂ - t₁)
            println("T:", Δt + t₂ - t₁, "r1",τr₁,"r2", τr₂)

            if( !(coupled_t & ref_coupled) )
                # Check if linval or modified for eff
                #v₁, v₂, ref_pos_coupled = lindvall_roger(x₁, x₂, τr₁, τr₂) 
                v₁, v₂, ref_pos_coupled = modified_lindvall_roger(x₁, x₂, τr₁, τr₂) # Try and couple velocities
            else
                # τr₁ = τr₂
                v₁, v₂, ref_pos_coupled = reflection_maximal(x₁, x₂, τr₁) 
            end
            v₁ = (v₁- x₁) ./ τr₁; v₂ = (v₂ - x₂) ./ τr₂; 

            grad_1, grad_2 = ∇U(x₁), ∇U(x₂) 
            println("x: ",x₁, "x2:", x₂, "r1",τr₁,"r2", τr₂)
            println("v: ",v₁)
            thin_1 = get_thin(v₁, grad_1, H,  0.0)
            thin_2 = get_thin(v₂, grad_2, H, Δt + t₂ - t₁)
            
            τb₁, τb₂, b_coupled = coupling_bounce(thin_1, thin_2)

            is_bounce₁ = τb₁ < τr₁
            is_bounce₂ = τb₂ < τr₂

            if(!(is_bounce₁ & is_bounce₂))
                coupled_v = coupled_x & ref_pos_coupled
            else
                coupled_v = false
            end
        end

        #println("ref ",τr₁, τr₂, " bnc ",τb₁, τb₂)
        τ₁ = min(τr₁, τb₁)
        τ₂ = min(τr₂, τb₂)

        next_event_info = coupled_event_info(τ₁, τ₂, is_bounce₁, is_bounce₂, ref_coupled, ref_pos_coupled, b_coupled)

        state_1 = (t₁, x₁, v₁)
        state_2 = (t₂, x₂, v₂)

        coupled = coupled_next
        coupled_next = coupled_t & coupled_x & coupled_v

        return state_1, state_2, next_event_info, coupled_next, coupled
    end

    return PDMP(init_coupling, kernel_event)
end

