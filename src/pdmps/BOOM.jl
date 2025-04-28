include("pdmp.jl")

"""
Coupled Boomerang kernel
"""

mutable struct BOOM_coupled_status <: coupled_info
    """ Coupled status for the current state of the sampler (not next event)
    Also store information such as when the process couples and number of gradient evaluations
    """
    coupled::Bool
    coupled_next::Bool
    coupled_t::Bool
    coupled_x::Bool
    coupled_v::Bool
    stoch_time::Float64

    num_grad_1::Int
    num_grad_2::Int
    num_grad_coupled::Int
    
    num_bounce_1::Int
    num_bounce_2::Int
    num_ref_1::Int
    num_ref_2::Int
    num_precoupled_1::Int
    num_precoupled_2::Int
end

struct next_event_coupled_BOOM
    """ Event information for next coupled BPS
    Next event times, bounce indicators, flags for coupled time, position and bounce
    """
    τ₁::Float64
    τ₂::Float64
    bounce₁::Bool
    bounce₂::Bool
    grad₁::Vector
    grad₂::Vector
    ref_coupled::Bool
    ref_pos_coupled::Bool
    b_coupled::Bool
end


function BOOM_coupling(∇U::Function, H, Σ, xstar::Vector, Δt::Float64, ΔM::Int, λᵣ::Float64, h_::Function = (x) -> 0., continuous::Bool = false, couple_mode::AbstractString = "antithetic")

    ## Calculate M1 and M2 to bound the Boomerang
    Σ_sqrt = sqrt(Σ)
    Σ_sqrt_inv = inv(Σ_sqrt)
    Σ_inv = inv(Σ)
    M1 = opnorm(H)
    grad_ref = ∇U(xstar)
    M2 = sqrt(dot(grad_ref,grad_ref))

    """ Estimator utility functions """

    function h(x, v, t)
        # Handelling different estimator types
        if continuous
            return h_(x, v, t)
        else
            return h_(x)
        end
    end
    
    function accumulate_h(h_val, x, v, t, time_seq, t_next)
        # Accumulate h over time_seq from t to t_next
        # Only use if no events from (x,v,t) to (x',v',t_next)

        if continuous
            h_val = map((u,v) -> (u + v / ΔM), h_val, h(x, v, t_next - t)) 
        else
            ind = findfirst( time_seq .> t)
            while ind <= length(time_seq) && t_next >= time_seq[ind] 
                x, v = dynamics(x, v, time_seq[ind]-t)
                h_t = h(x, v, time_seq[ind]-t)
                h_val = map((u,v) -> (u + v / ΔM), h_val, h_t) 
                ind += 1
            end
        end
        return h_val
    end

    function get_thin(x::Vector, v::Vector, grad::Vector, shift::Float64)
        # Set the thinning bound (taken from ICML Boom)

        phaseSpaceNorm = sqrt(dot(x-xstar,x-xstar) + dot(v,v))
        a = M1 * phaseSpaceNorm^2 + M2 * phaseSpaceNorm
        b = v'*grad

        return AffinePoisson(a, b, 0.0, shift)
    end

    """ PDMP specific functions """

    function dynamics(x, v, t)
        # Gaussian Dynamics
        x_new = xstar + (x-xstar)*cos(t) + v*sin(t)
        v_new = -(x - xstar)*sin(t) + v*cos(t)
        return x_new, v_new
    end
    
    function bounce_thinning(v, grad, thin, τ, rand_unif = rand())
        # Apply the bounce kernel with thinning
        switch_rate = v'*grad
        upper_bound = rate(thin, τ)
        if upper_bound  < switch_rate - 1e-8
            println("upper bound", upper_bound)
            println("actual rate", switch_rate)
        end
        if rand_unif * upper_bound <= switch_rate
            sk_grad = Σ_sqrt'*grad
            return v - 2 * switch_rate / dot(sk_grad,sk_grad) * Σ_sqrt * sk_grad
        else 
            return v
        end
    end

    function kernel(state, time_seq, h_val)
        # BOOM kernel move state along time_seq accumulating h_val
        # Return new state, h_val and number of gradient evals

        num_event = 0
        n_ind = length(time_seq)
        t, x, v, thin = state

        # Get next bounce
        τr, τb = -log(rand()) / λᵣ, rand(thin)
        τb = τb - thin.shift 
        is_bounce = τb < τr
        τ = min(τr, τb)

        t_next = t + τ
        
        for i in 1:n_ind
            # Accumulate/move process untill next event > next time on time_seq
            while(t_next < time_seq[i])
                h_val = accumulate_h(h_val, x, v, t, time_seq, t_next)
                x, v = dynamics(x, v, t_next-t)
                t = t_next
                if is_bounce
                    num_event += 1
                    grad = ∇U(x) - Σ_inv * (x - xstar)
                    v = bounce_thinning(v, grad, thin, τ)
                else
                    num_event += 1
                    grad = ∇U(x) - Σ_inv * (x - xstar)
                    v = Σ_sqrt * randn(length(v))
                end
                # Get next bounce
                thin = get_thin(x, v, grad, thin.shift)
                τr, τb = -log(rand()) / λᵣ, rand(thin)
                τb = τb - thin.shift 
                is_bounce = τb < τr
                τ = min(τr, τb)
                t_next = t + τ
            end
            # Accumulate/move process to time_seq[i] 
            h_val = accumulate_h(h_val, x, v, t, time_seq, time_seq[i])
            x, v = dynamics(x, v, time_seq[i]-t)
            thin = update_thin(thin, time_seq[i] - t) 
            t = time_seq[i]
            τ = t_next-t
        end

        # Update linear thinning to be valid at final time
        new_state = (t, x, v, thin)
        return new_state, h_val, num_event
    end

    function init(x₁::Vector, x₂::Vector)
        # BPS Init coupled sampler 
        # Progress Z1 by Δ

        t₁ = 0.0; t₂ = 0.0
        
        # Try and couple velocities
        τr₁, τr₂, ref_coupled = coupling_refresh(λᵣ, Δt + t₂ - t₁, couple_mode)
        v₁ = Σ_sqrt*randn(length(x₁)); v₂ = Σ_sqrt*randn(length(x₁)); 
        ref_pos_coupled = false

        grad_1, grad_2 = ∇U(x₁) - Σ_inv * (x₁ - xstar), ∇U(x₂) - Σ_inv * (x₂ - xstar)
        thin_1 = get_thin(x₁, v₁, grad_1, 0.0)
        thin_2 = get_thin(x₂, v₂, grad_2, Δt + t₂ - t₁)

        # Compute next times
        τb₁, τb₂, b_coupled = coupling_bounce(thin_1, thin_2)
        
        τ₁ = min(τr₁, τb₁)
        τ₂ = min(τr₂, τb₂)

        is_bounce₁ = τb₁ < τr₁
        is_bounce₂ = τb₂ < τr₂

        next_event_info = coupled_event_info(τ₁, τ₂, is_bounce₁, is_bounce₂, ref_coupled, ref_pos_coupled, b_coupled)

        state_1 = (t₁, x₁, v₁, thin_1)
        state_2 = (t₂, x₂, v₂, thin_2)
        
        return state_1, state_2, next_event_info, false, false
    end

    function coupled_kernel(state_1, state_2, next_event_info::coupled_event_info, coupled_next, coupled)

        t₁, x₁, v₁, thin_1 = state_1
        t₂, x₂, v₂, thin_2 = state_2

        τ₁, τ₂ = next_event_info.τ₁, next_event_info.τ₂

        if next_event_info.bounce₁ & next_event_info.bounce₂
            
            coupled_t = next_event_info.b_coupled
            coupled_x = false; coupled_v = false
            ref_pos_coupled = false

            t₁ += τ₁
            t₂ += τ₂

            x₁, v₁ = dynamics(x₁, v₁, τ₁)
            x₂, v₂ = dynamics(x₂, v₂, τ₂)

            grad_1, grad_2 = ∇U(x₁) - Σ_inv * (x₁ - xstar), ∇U(x₂) - Σ_inv * (x₂ - xstar)

            common_u = rand()
            v₁ = bounce_thinning(v₁, grad_1, thin_1, τ₁, common_u)
            v₂ = bounce_thinning(v₂, grad_2, thin_2, τ₂, common_u)
            
            thin_1 = get_thin(x₁, v₁, grad_1, 0.0)
            thin_2 = get_thin(x₂, v₂, grad_2, Δt + t₂ - t₁)
            
            τb₁, τb₂, b_coupled = coupling_bounce(thin_1, thin_2)
            τr₁, τr₂, ref_coupled = coupling_refresh(λᵣ, Δt + t₂ - t₁, couple_mode)

            is_bounce₁ = τb₁ < τr₁
            is_bounce₂ = τb₂ < τr₂

        elseif next_event_info.bounce₁
            
            coupled_t = false; coupled_x = false; coupled_v = false
            ref_pos_coupled = false

            t₁ += τ₁
            t₂ += τ₂

            x₁, v₁ = dynamics(x₁, v₁, τ₁)
            x₂, v₂ = dynamics(x₂, v₂, τ₂)

            grad_1, grad_2 = ∇U(x₁) - Σ_inv * (x₁ - xstar), ∇U(x₂) - Σ_inv * (x₂ - xstar)
            v₁ = bounce_thinning(v₁, grad_1, thin_1, τ₁)
            v₂ = Σ_sqrt*randn(length(v₂))

            thin_1 = get_thin(x₁, v₁, grad_1, 0.0)
            thin_2 = get_thin(x₂, v₂, grad_2, Δt + t₂ - t₁)
            
            τb₁, τb₂, b_coupled = coupling_bounce(thin_1, thin_2)
            τr₁, τr₂, ref_coupled = coupling_refresh(λᵣ, Δt + t₂ - t₁, couple_mode)

            is_bounce₁ = τb₁ < τr₁
            is_bounce₂ = τb₂ < τr₂

        elseif next_event_info.bounce₂
            
            coupled_t = false; coupled_x = false; coupled_v = false
            ref_pos_coupled = false

            t₁ += τ₁
            t₂ += τ₂

            x₁, v₁ = dynamics(x₁, v₁, τ₁)
            x₂, v₂ = dynamics(x₂, v₂, τ₂)

            grad_1, grad_2 = ∇U(x₁) - Σ_inv * (x₁ - xstar), ∇U(x₂) - Σ_inv * (x₂ - xstar)
            v₂ = bounce_thinning(v₂, grad_2, thin_2, τ₂)
            v₁ = Σ_sqrt*randn(length(v₁))
            
            thin_1 = get_thin(x₁, v₁, grad_1, 0.0)
            thin_2 = get_thin(x₂, v₂, grad_2, Δt + t₂ - t₁)
            
            τb₁, τb₂, b_coupled = coupling_bounce(thin_1, thin_2)
            τr₁, τr₂, ref_coupled = coupling_refresh(λᵣ, Δt + t₂ - t₁, couple_mode)
            
            is_bounce₁ = τb₁ < τr₁
            is_bounce₂ = τb₂ < τr₂

        else
            
            coupled_t = next_event_info.ref_coupled 
            coupled_x = next_event_info.ref_pos_coupled

            t₁ += τ₁
            t₂ += τ₂

            x₁, v₁ = dynamics(x₁, v₁, τ₁)
            x₂, v₂ = dynamics(x₂, v₂, τ₂)

            # Update the refreshment PRIOR to the velocity!
            τr₁, τr₂, ref_coupled = coupling_refresh(λᵣ, Δt + t₂ - t₁, couple_mode)

            if( !(coupled_t & ref_coupled) )
                # Check if linval or modified for eff
                v₁, v₂, ref_pos_coupled = lindvall_roger(x₁*cos(τr₁), x₂*cos(τr₂), abs(sin(τr₁))*Σ_sqrt, Σ_sqrt_inv/abs(sin(τr₁)), abs(sin(τr₂))*Σ_sqrt) 
            else
                # τr₁ = τr₂
                v₁, v₂, ref_pos_coupled = reflection_maximal(x₁*cos(τr₁), x₂*cos(τr₂), abs(sin(τr₁))*Σ_sqrt, Σ_sqrt_inv/abs(sin(τr₁)), abs(sin(τr₂))*Σ_sqrt)                                 
            end
            v₁ = (v₁ - x₁ .* cos(τr₁)) ./ sin(τr₁)
            v₂ = (v₂ - x₂ .* cos(τr₂)) ./ sin(τr₂)

            grad_1, grad_2 = ∇U(x₁) - Σ_inv * (x₁ - xstar), ∇U(x₂) - Σ_inv * (x₂ - xstar)
            thin_1 = get_thin(x₁, v₁, grad_1, 0.0)
            thin_2 = get_thin(x₂, v₂, grad_2, Δt + t₂ - t₁)
            
            τb₁, τb₂, b_coupled = coupling_bounce(thin_1, thin_2)

            is_bounce₁ = τb₁ < τr₁
            is_bounce₂ = τb₂ < τr₂

            if !(is_bounce₁ & is_bounce₂)
                coupled_v = coupled_x & ref_pos_coupled
            else
                coupled_v = false
            end
        end

        τ₁ = min(τr₁, τb₁)
        τ₂ = min(τr₂, τb₂)

        next_event_info = coupled_event_info(τ₁, τ₂, is_bounce₁, is_bounce₂, ref_coupled, ref_pos_coupled, b_coupled)

        state_1 = (t₁, x₁, v₁, thin_1)
        state_2 = (t₂, x₂, v₂, thin_2)
        
        if(!coupled)
            coupled = coupled_next
            coupled_next = coupled_t & coupled_x & coupled_v
        end

        return state_1, state_2, next_event_info, coupled_next, coupled
    end

    return coupled_pdmp(init, kernel, coupled_kernel, dynamics, λᵣ, Δt, ΔM, accumulate_h, h, couple_mode)
end