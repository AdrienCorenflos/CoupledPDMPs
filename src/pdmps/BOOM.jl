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
    
    function accumulate_h(h_val, z, time_seq, t_next)
        # Accumulate h over time_seq from t to t_next
        # Only use if no events from (x,v,t) to (x',v',t_next)
        t, x, v = z
        # Init
        if h_val == undef
            h_val = 0.0 .* h(x, v, 0.0)
        end

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

    """ PDMP specific functions """

    function dynamics(x, v, t)
        # Gaussian Dynamics
        x_new = xstar + (x-xstar)*cos(t) + v*sin(t)
        v_new = -(x - xstar)*sin(t) + v*cos(t)
        return x_new, v_new
    end

    function bounce_prob(v, grad, thin, τ)
        # Apply the bounce kernel with thinning
        switch_rate = v'*grad
        upper_bound = rate(thin, τ)
        if upper_bound  < switch_rate - 1e-8
            println("upper bound", upper_bound)
            println("actual rate", switch_rate)
        end
        return max(0.0, switch_rate/upper_bound)
    end

    function get_thin_bound(x::Vector, v::Vector, grad::Vector, shift::Float64)
        # Set the thinning bound (taken from ICML Boom)

        phaseSpaceNorm = sqrt(dot(x-xstar,x-xstar) + dot(v,v))
        a = M1 * phaseSpaceNorm^2 + M2 * phaseSpaceNorm
        b = v'*grad

        return AffinePoisson(a, b, 0.0, shift)
    end

    function get_thin(z, τr)
        # Get the next bounce event using thinning
        # return event time, number grads used, final gradient
        t, x, v, thin = z
        num_grad = 0
        grad = zeros(length(x))

        τb = rand(thin)
        τb = τb - thin.shift
        τ = τb
        while true
            if τ > τr
                return τ, grad, num_grad
            end

            x, v = dynamics(x, v, τb)
            grad = ∇U(x) - Σ_inv * (x - xstar); num_grad += 1

            if rand() < bounce_prob(v, grad, thin, τb)
                return τ, grad, num_grad
            end
            thin = get_thin_bound(x, v, grad, 0.0)
            τb = rand(thin)
            τ += τb
        end
    end

    function coupling_bounce(z₁, z₂, τr₁, τr₂, time_shift)

        t₁, x₁, v₁, thin_1 = z₁
        t₂, x₂, v₂, thin_2 = z₂
        num_grad_1 = 0
        num_grad_2 = 0

        # Store the shift for coupling in time
        τb₁, τb₂, coupled = thorisson(thin_1, thin_2)
        τb₂ = τb₂ - thin_2.shift; τb₁ = τb₁ - thin_1.shift
        τ₁ = τb₁;  τ₂ = τb₂
        
        grad₁, grad₂ = zeros(length(x₁)), zeros(length(x₁))

        while true

            # Check if possible to stop sampling on processes
            # stop if next bounce after refreshment
            stop_1, stop_2 = τ₁ > τr₁, τ₂ > τr₂

            if !stop_1
                x₁_, v₁_ = dynamics(x₁, v₁, τ₁)
                grad₁ = ∇U(x₁_) - Σ_inv * (x₁_ - xstar)
                num_grad_1 += 1
                a₁ = bounce_prob(v₁_, grad₁, thin_1, τb₁)
            else
                coupled = false
                a₁ = 1.0
            end

            if !stop_2
                x₂_, v₂_ = dynamics(x₂, v₂, τ₂)
                grad₂ = ∇U(x₂_) - Σ_inv * (x₂_ - xstar)
                num_grad_2 += 1
                a₂ = bounce_prob(v₂_, grad₂, thin_2, τb₂)
            else
                coupled = false
                a₂ = 1.0
            end

            # Propose next bounce event time
            u = rand()
            if u < min(a₁, a₂) 
                return τ₁, τ₂, coupled, num_grad_1, num_grad_2, grad₁, grad₂

            elseif u < a₁ 
                # Continue thinning process 2 
                thin_2 = get_thin_bound(x₂_, v₂_, grad₂, 0.0)
                z = (t₂ + τ₂, x₂_, v₂_, thin_2)
                τb, grad₂, num_grad_bounce = get_thin(z, τr₂)
                τ₂ += τb; num_grad_2 += num_grad_bounce
                return τ₁, τ₂, false, num_grad_1, num_grad_2, grad₁, grad₂
                
            elseif u < a₂ 
                # Continue thinning process 1
                thin_1 = get_thin_bound(x₁_, v₁_, grad₁, 0.0)
                z = (t₁ + τ₁, x₁_, v₁_, thin_1)
                τb, grad₁, num_grad_bounce = get_thin(z, τr₁)
                τ₁ += τb; num_grad_1 += num_grad_bounce

                return τ₁, τ₂, false, num_grad_1, num_grad_2, grad₁, grad₂
            end
            
            # If u > a₁ and a₂ then continue coupled thinning
            thin_1 = get_thin_bound(x₁_, v₁_, grad₁, 0.0)
            thin_2 = get_thin_bound(x₂_, v₂_, grad₂, time_shift + τ₂ - τ₁) 

            τb₁, τb₂, coupled = thorisson(thin_1, thin_2)
            τb₂ = τb₂ - thin_2.shift; τb₁ = τb₁ - thin_1.shift
            τ₁ += τb₁;  τ₂ += τb₂
        end
    end

    function get_next_event(z₁, z₂, coupled_status)
        # Get information for next coupled event, also return gradient evaluations required
        t₁, _ = z₁
        t₂, _ = z₂

        if coupled_status.coupled_t
            time_shift = 0.0
        else
            time_shift = Δt + t₂ - t₁
        end

        τr₁, τr₂, ref_coupled = coupling_refresh(λᵣ, Δt + t₂ - t₁, couple_mode) 

        τb₁, τb₂, b_coupled, num_grad_1, num_grad_2, grad₁, grad₂ = coupling_bounce(z₁, z₂, τr₁, τr₂, time_shift)
        if coupled_status.coupled
            coupled_status.num_grad_coupled += num_grad_1
        else
            coupled_status.num_grad_1 += num_grad_1
            coupled_status.num_grad_2 += num_grad_2
        end
        
        is_bounce₁ = τb₁ < τr₁
        is_bounce₂ = τb₂ < τr₂

        τ₁ = min(τr₁, τb₁)
        τ₂ = min(τr₂, τb₂)

        next_event = next_event_coupled_BOOM(τ₁, τ₂, is_bounce₁, is_bounce₂, grad₁, grad₂, ref_coupled, coupled_status.coupled_x, b_coupled)

        return next_event, coupled_status
    end

    function kernel(state, time_seq, h_val, coupled_status = undef, next_event_info = undef, process = 1)
        # BOOM kernel move state along time_seq accumulating h_val
        # Return new state, h_val and number of gradient evals

        num_grad = 0
        num_e = 0
        num_r = 0
        n_ind = length(time_seq)
        t, x, v, thin = state

        # Get next bounce
        if !(next_event_info == undef)
            τ = process == 1 ? next_event_info.τ₁ : next_event_info.τ₂
            is_bounce = process == 1 ? next_event_info.bounce₁ : next_event_info.bounce₂
            grad = process == 1 ? next_event_info.grad₁ : next_event_info.grad₂
        else
            τr = -log(rand()) / λᵣ
            τb, grad, num_grad_bounce = get_thin(state, τr)
            num_grad += num_grad_bounce
            is_bounce = τb < τr
            τ = min(τr, τb)
        end

        t_next = t + τ
        
        for i in 1:n_ind
            # Accumulate/move process untill next event > next time on time_seq
            while(t_next < time_seq[i])
                h_val = accumulate_h(h_val, (t, x, v), time_seq, t_next)
                x, v = dynamics(x, v, t_next-t)
                t = t_next
                if is_bounce
                    v = bounce(v, grad, Σ_sqrt)
                    num_e += 1
                else
                    num_r += 1
                    num_grad += 1
                    grad = ∇U(x) - Σ_inv * (x - xstar)
                    v = Σ_sqrt * randn(length(v))
                end
                # Get next bounce
                τr = -log(rand()) / λᵣ
                thin = get_thin_bound(x, v, grad, thin.shift)
                z = (t, x, v, thin)
                τb, grad, num_grad_bounce = get_thin(z, τr)
                num_grad += num_grad_bounce
                
                is_bounce = τb < τr
                τ = min(τr, τb)
                t_next = t + τ
            end
            # Accumulate/move process to time_seq[i] 
            h_val = accumulate_h(h_val, (t, x, v), time_seq, time_seq[i])
            x, v = dynamics(x, v, time_seq[i]-t)
            thin = update_thin(thin, time_seq[i] - t) 
            t = time_seq[i]
            τ = t_next-t
        end

        if !(coupled_status == undef)
            coupled_status.coupled_x = false
            coupled_status.coupled_v = false
            if process == 1
                coupled_status.num_grad_1 += num_grad
                coupled_status.num_bounce_1 += num_e
                coupled_status.num_ref_1 += num_r
            else
                coupled_status.num_grad_2 += num_grad
                coupled_status.num_bounce_2 += num_e
                coupled_status.num_ref_2 += num_r
            end
        end

        # Update linear thinning to be valid at final time
        new_state = (t, x, v, thin)
        return new_state, h_val, coupled_status
    end

    function init(x₁::Vector, x₂::Vector)
        # BOOM Init coupled sampler 
        # Progress Z1 by Δ

        t₁ = 0.0; t₂ = 0.0
        v₁ = Σ_sqrt*randn(length(x₁)); v₂ = Σ_sqrt*randn(length(x₁)); 
        
        grad_1, grad_2 = ∇U(x₁) - Σ_inv * (x₁ - xstar), ∇U(x₂) - Σ_inv * (x₂ - xstar)
        thin_1 = get_thin_bound(x₁, v₁, grad_1, 0.0)
        thin_2 = get_thin_bound(x₂, v₂, grad_2, Δt + t₂ - t₁)

        z_1 = (t₁, x₁, v₁, thin_1)
        z_2 = (t₂, x₂, v₂, thin_2)

        # Get next event times
        coupled_status = BOOM_coupled_status(false, false, false, false, false, Inf, 0, 0, 0,    0, 0, 0, 0, 0, 0)
        next_event_info, coupled_status = get_next_event(z_1, z_2, coupled_status)
        
        return z_1, z_2, next_event_info, coupled_status, false
    end

    function coupled_kernel(z_1, z_2, next_event_info::next_event_coupled_BOOM, coupled_status::BOOM_coupled_status, coupled)

        t₁, x₁, v₁, thin_1 = z_1
        t₂, x₂, v₂, thin_2 = z_2

        τ₁, τ₂ = next_event_info.τ₁, next_event_info.τ₂

        if next_event_info.bounce₁ & next_event_info.bounce₂
            coupled_status.coupled_t = next_event_info.b_coupled
            if !next_event_info.b_coupled
                coupled_status.coupled_x = false
                coupled_status.coupled_v = false
            end
            
            coupled_status.num_bounce_1 += 1;    coupled_status.num_bounce_2 += 1

            t₁ += τ₁
            t₂ += τ₂

            x₁, v₁ = dynamics(x₁, v₁, τ₁)
            x₂, v₂ = dynamics(x₂, v₂, τ₂)

            v₁ = bounce(v₁, next_event_info.grad₁, Σ_sqrt)
            v₂ = bounce(v₂, next_event_info.grad₂, Σ_sqrt)

            if coupled_status.coupled_t
                time_shift = 0.0
            else
                time_shift = Δt + t₂ - t₁
            end

            thin_1 = get_thin_bound(x₁, v₁, next_event_info.grad₁, 0.0)
            thin_2 = get_thin_bound(x₂, v₂, next_event_info.grad₂, Δt + t₂ - t₁)

            z_1 = (t₁, x₁, v₁, thin_1)
            z_2 = (t₂, x₂, v₂, thin_2)

            next_event_info, coupled_status = get_next_event(z_1, z_2, coupled_status)

        elseif next_event_info.bounce₁
            coupled_status.coupled_t = false; coupled_status.coupled_x = false
            coupled_status.coupled_v = false
            
            coupled_status.num_bounce_1 += 1;    coupled_status.num_ref_1 += 1

            t₁ += τ₁
            t₂ += τ₂

            x₁, v₁ = dynamics(x₁, v₁, τ₁)
            x₂, v₂ = dynamics(x₂, v₂, τ₂)

            v₁ = bounce(v₁, next_event_info.grad₁, Σ_sqrt)
            v₂ = Σ_sqrt*randn(length(v₂))

            thin_1 = get_thin_bound(x₁, v₁, next_event_info.grad₁, 0.0)
            thin_2 = get_thin_bound(x₂, v₂, ∇U(x₂) - Σ_inv * (x₂ - xstar), Δt + t₂ - t₁)
            coupled_status.num_grad_2 += 1
            
            z_1 = (t₁, x₁, v₁, thin_1)
            z_2 = (t₂, x₂, v₂, thin_2)

            next_event_info, coupled_status = get_next_event(z_1, z_2, coupled_status)

        elseif next_event_info.bounce₂
            coupled_status.coupled_t = false; coupled_status.coupled_x = false
            coupled_status.coupled_v = false

            coupled_status.num_bounce_2 += 1;            coupled_status.num_ref_1 += 1

            t₁ += τ₁
            t₂ += τ₂

            x₁, v₁ = dynamics(x₁, v₁, τ₁)
            x₂, v₂ = dynamics(x₂, v₂, τ₂)

            v₂ = bounce(v₂, next_event_info.grad₂, Σ_sqrt)
            v₁ = Σ_sqrt*randn(length(v₁))
            
            thin_1 = get_thin_bound(x₁, v₁, ∇U(x₁) - Σ_inv * (x₁ - xstar), 0.0)
            coupled_status.num_grad_1 += 1
            thin_2 = get_thin_bound(x₂, v₂, next_event_info.grad₂, Δt + t₂ - t₁)
            
            z_1 = (t₁, x₁, v₁, thin_1)
            z_2 = (t₂, x₂, v₂, thin_2)

            next_event_info, coupled_status = get_next_event(z_1, z_2, coupled_status)

        else
            coupled_status.coupled_t = next_event_info.ref_coupled
            coupled_status.coupled_x = next_event_info.ref_pos_coupled

            coupled_status.num_ref_1 += 1;    coupled_status.num_ref_2 += 1

            t₁ += τ₁
            t₂ += τ₂

            x₁, v₁ = dynamics(x₁, v₁, τ₁)
            x₂, v₂ = dynamics(x₂, v₂, τ₂)

            if coupled_status.coupled_t
                time_shift = 0.0
            else
                time_shift = Δt + t₂ - t₁
            end

            # Update the refreshment PRIOR to the velocity!
            τr₁, τr₂, ref_coupled = coupling_refresh(λᵣ, Δt + t₂ - t₁, couple_mode)

            if coupled_status.coupled_t & ref_coupled
                # τr₁ = τr₂
                v₁, v₂, ref_pos_coupled = reflection_maximal(x₁*cos(τr₁), x₂*cos(τr₂), abs(sin(τr₁))*Σ_sqrt, Σ_sqrt_inv/abs(sin(τr₁)), abs(sin(τr₂))*Σ_sqrt)
            else
                # Check if linval or modified for eff
                v₁, v₂, ref_pos_coupled = lindvall_roger(x₁*cos(τr₁), x₂*cos(τr₂), abs(sin(τr₁))*Σ_sqrt, Σ_sqrt_inv/abs(sin(τr₁)), abs(sin(τr₂))*Σ_sqrt) 
            end
            v₁ = (v₁ - x₁ .* cos(τr₁)) ./ sin(τr₁)
            v₂ = (v₂ - x₂ .* cos(τr₂)) ./ sin(τr₂)

            grad_1, grad_2 = ∇U(x₁) - Σ_inv * (x₁ - xstar), ∇U(x₂) - Σ_inv * (x₂ - xstar)
            
            if coupled
                coupled_status.num_grad_coupled += 1
            else
                coupled_status.num_grad_1 += 1
                coupled_status.num_grad_2 += 1
            end

            thin_1 = get_thin_bound(x₁, v₁, grad_1, 0.0)
            thin_2 = get_thin_bound(x₂, v₂, grad_2, Δt + t₂ - t₁)

            z_1 = (t₁, x₁, v₁, thin_1)
            z_2 = (t₂, x₂, v₂, thin_2)
            
            # Get the next bounce event times
            τb₁, τb₂, b_coupled, num_grad_1, num_grad_2, grad₁, grad₂ = coupling_bounce(z_1, z_2, τr₁, τr₂, time_shift)

            if coupled
                coupled_status.num_grad_coupled += num_grad_1
            else
                coupled_status.num_grad_1 += num_grad_1
                coupled_status.num_grad_2 += num_grad_2
            end

            is_bounce₁ = τb₁ < τr₁
            is_bounce₂ = τb₂ < τr₂

            coupled_status.coupled_v = coupled_status.coupled_x & ref_pos_coupled
            τ₁ = min(τr₁, τb₁)
            τ₂ = min(τr₂, τb₂)
            next_event_info = next_event_coupled_BOOM(τ₁, τ₂, is_bounce₁, is_bounce₂, grad₁, grad₂, ref_coupled, ref_pos_coupled, b_coupled)
        end
        
        if !coupled
            coupled = coupled_status.coupled_next
            coupled_status.coupled = coupled
            coupled_status.coupled_next = coupled_status.coupled_t & coupled_status.coupled_x & coupled_status.coupled_v
            if coupled
                coupled_status.stoch_time = t₁
                coupled_status.num_precoupled_1 = coupled_status.num_bounce_1 + coupled_status.num_ref_1
                coupled_status.num_precoupled_2 = coupled_status.num_bounce_2 + coupled_status.num_ref_2
            end
        end

        return z_1, z_2, next_event_info, coupled_status, coupled
    end

    return coupled_pdmp(init, kernel, coupled_kernel, dynamics, λᵣ, Δt, ΔM, accumulate_h, get_next_event)
end