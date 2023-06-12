include("pdmp.jl")
"""
Clean version of BPS_coupled sampler code...
"""
#############################################
function BOOM_coupling(∇U::Function, H::Matrix, Σ::Matrix, xstar::Vector, Δt::Float64, λᵣ::Float64, couple_mode::AbstractString = "independent")

    ## Calc M1 and M2
    Σ_sqrt = sqrt(Σ)
    Σ_sqrt_inv = inv(Σ_sqrt)
    Σ_inv = inv(Σ)
    M1 = opnorm(H)
    grad_ref = ∇U(xstar)
    M2 = sqrt(dot(grad_ref,grad_ref))

    function get_thin(x::Vector, v::Vector, grad::Vector, shift::Float64)
        
        # Set the thinning bound (taken from ICML Boom)
        phaseSpaceNorm = sqrt(dot(x-xstar,x-xstar) + dot(v,v))
        a = M1 * phaseSpaceNorm^2 + M2 * phaseSpaceNorm
        b = v'*grad

        return AffinePoisson(a, b, 0.0, shift)
    end

    # linear Dynamics
    function dynamics(x, v, t)
        x_new = xstar + (x-xstar)*cos(t) + v*sin(t)
        v_new = -(x - xstar)*sin(t) + v*cos(t)
        return x_new, v_new
    end

    ## Update v based on thinning
    function thinning_update_v(v, grad, thin, τ, rand_unif = rand())
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


    # BPS kernel Moves the state until stochastic time Tmax
    function kernel(state, T_seq, M, h_val, h = (x) -> 0.)
        t, x, v, thin = state
        τr, τb = -log(rand()) / λᵣ, rand(thin)
        τb = τb - thin.shift
        is_bounce = τb < τr
        τ = min(τr, τb)
        t_next = t + τ
        num_event = 1
        n_ind = length(T_seq)
        for i in 1:n_ind
            while(t_next < T_seq[i])
                x, v = dynamics(x, v, τ)
                t = t_next
                if(is_bounce)
                    num_event += 1
                    grad = ∇U(x) - Σ_inv * (x - xstar)
                    v = update_v(v, grad, rate(thin, τ))
                else
                    num_event += 1
                    grad = ∇U(x) - Σ_inv * (x - xstar)
                    v = Σ_sqrt * randn(length(v))
                end
                thin = get_thin(x, v, grad, 0.0)
                τr, τb = -log(rand()) / λᵣ, rand(thin)
                τb = τb - thin.shift
                is_bounce = τb < τr
                τ = min(τr, τb)
                t_next = t + τ
            end
            if h_val == undef
                x_t, _ = dynamics(x, v, T_seq[i] - t)
                h_val = map((u,v) -> (u + v / M), 0., h(x_t))
            else
                x_t, _ = dynamics(x, v, T_seq[i] - t)
                h_val = map((u,v) -> (u + v / M), h_val, h(x_t)) #accumulate h
            end
        end

        thin = update_thin(thin, T_seq[n_ind] - t)
        x, v = dynamics(x, v, T_seq[n_ind] - t)
        t = T_seq[n_ind]
        new_state = (t, x, v, thin)
        return new_state, h_val, num_event
    end

    function init(x₁::Vector, x₂::Vector)

        t₁ = 0.0; t₂ = 0.0
        
        τr₁, τr₂, ref_coupled = coupling_refresh(λᵣ, Δt + t₂ - t₁, couple_mode)
        
        v₁ = Σ_sqrt*randn(length(x₁)); v₂ = Σ_sqrt*randn(length(x₁)); 
        ref_pos_coupled = false

        thin_1 = get_thin(x₁, v₁, ∇U(x₁), 0.0)
        thin_2 = get_thin(x₂, v₂, ∇U(x₂), Δt + t₂ - t₁)

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

        if(next_event_info.bounce₁ & next_event_info.bounce₂)
            coupled_t = next_event_info.b_coupled
            coupled_x = false; coupled_v = false
            ref_pos_coupled = false

            t₁ += next_event_info.τ₁
            t₂ += next_event_info.τ₂

            x₁, v₁ = dynamics(x₁, v₁, next_event_info.τ₁)
            x₂, v₂ = dynamics(x₂, v₂, next_event_info.τ₂)

            grad_1, grad_2 = ∇U(x₁) - Σ_inv * (x₁ - xstar), ∇U(x₂) - Σ_inv * (x₂ - xstar)
            common_u = rand()
            v₁ = thinning_update_v(v₁, grad_1, thin_1, τ₁, common_u)
            v₂ = thinning_update_v(v₂, grad_2, thin_2, τ₂, common_u)
            
            # Compute next event times. We can do this after the event given they are only bounces
            thin_1 = get_thin(x₁, v₁, grad_1, 0.0)
            thin_2 = get_thin(x₂, v₂, grad_2, Δt + t₂ - t₁)
            
            τb₁, τb₂, b_coupled = coupling_bounce(thin_1, thin_2)
            τr₁, τr₂, ref_coupled = coupling_refresh(λᵣ, Δt + t₂ - t₁, couple_mode)

            is_bounce₁ = τb₁ < τr₁
            is_bounce₂ = τb₂ < τr₂

        elseif(next_event_info.bounce₁)
            
            coupled_t = false; coupled_x = false; coupled_v = false
            ref_pos_coupled = false

            t₁ += next_event_info.τ₁
            t₂ += next_event_info.τ₂

            x₁, v₁ = dynamics(x₁, v₁, next_event_info.τ₁)
            x₂, v₂ = dynamics(x₂, v₂, next_event_info.τ₂)

            grad_1, grad_2 = ∇U(x₁) - Σ_inv * (x₁ - xstar), ∇U(x₂) - Σ_inv * (x₂ - xstar)
            v₁ = update_v(v₁, grad_1, rate(thin_1, next_event_info.τ₁))
            v₂ = randn(length(v₂))

            # Compute next event times. We can do this after the event given they are only bounces
            thin_1 = get_thin(x₁, v₁, grad_1, 0.0)
            thin_2 = get_thin(x₂, v₂, grad_2, Δt + t₂ - t₁)
            
            τb₁, τb₂, b_coupled = coupling_bounce(thin_1, thin_2)
            τr₁, τr₂, ref_coupled = coupling_refresh(λᵣ, Δt + t₂ - t₁, couple_mode)

            is_bounce₁ = τb₁ < τr₁
            is_bounce₂ = τb₂ < τr₂

        elseif(next_event_info.bounce₂)
            #println("3")
            coupled_t = false; coupled_x = false; coupled_v = false
            ref_pos_coupled = false

            t₁ += next_event_info.τ₁
            t₂ += next_event_info.τ₂

            x₁, v₁ = dynamics(x₁, v₁, next_event_info.τ₁)
            x₂, v₂ = dynamics(x₂, v₂, next_event_info.τ₂)

            grad_1, grad_2 = ∇U(x₁) - Σ_inv * (x₁ - xstar), ∇U(x₂) - Σ_inv * (x₂ - xstar)
            v₂ = update_v(v₂, grad_2, rate(thin_2, next_event_info.τ₂))
            v₁ = randn(length(v₁))
            
            # Compute next event times. We can do this after the event given they are only bounces
            thin_1 = get_thin(x₁, v₁, grad_1, 0.0)
            thin_2 = get_thin(x₂, v₂, grad_2, Δt + t₂ - t₁)
            
            τb₁, τb₂, b_coupled = coupling_bounce(thin_1, thin_2)
            τr₁, τr₂, ref_coupled = coupling_refresh(λᵣ, Δt + t₂ - t₁, couple_mode)
            
            is_bounce₁ = τb₁ < τr₁
            is_bounce₂ = τb₂ < τr₂

        else
            #println("4")
            coupled_t = next_event_info.ref_coupled 
            coupled_x = next_event_info.ref_pos_coupled

            t₁ += next_event_info.τ₁
            t₂ += next_event_info.τ₂

            x₁, v₁ = dynamics(x₁, v₁, next_event_info.τ₁)
            x₂, v₂ = dynamics(x₂, v₂, next_event_info.τ₂)

            # Update the refreshment PRIOR to the velocity!
            τr₁, τr₂, ref_coupled = coupling_refresh(λᵣ, Δt + t₂ - t₁, couple_mode)

            if( !(coupled_t & ref_coupled) )
                # Check if linval or modified for eff
                v₁, v₂, ref_pos_coupled = lindvall_roger(x₁*cos(τr₁), x₂*cos(τr₂), abs(sin(τr₁))*Σ_sqrt, Σ_sqrt_inv/abs(sin(τr₁)), abs(sin(τr₂))*Σ_sqrt) 
                #lindvall_roger(mx, my, Q_x, Q_x_inv, Q_y)
            else
                # τr₁ = τr₂ ### Replace with reflex maximal
                v₁, v₂, ref_pos_coupled = lindvall_roger(x₁*cos(τr₁), x₂*cos(τr₂), abs(sin(τr₁))*Σ_sqrt, Σ_sqrt_inv/abs(sin(τr₁)), abs(sin(τr₂))*Σ_sqrt)                 
                println("ref",ref_pos_coupled)
                #v₁, v₂, ref_pos_coupled = reflection_maximal(x₁, x₂, τr₁) 
            end
            v₁ = (v₁ - x₁ .* cos(τr₁)) ./ sin(τr₁)
            v₂ = (v₂ - x₂ .* cos(τr₂)) ./ sin(τr₂)

            grad_1, grad_2 = ∇U(x₁) - Σ_inv * (x₁ - xstar), ∇U(x₂) - Σ_inv * (x₂ - xstar)
            thin_1 = get_thin(x₁, v₁, grad_1, 0.0)
            thin_2 = get_thin(x₂, v₂, grad_2, Δt + t₂ - t₁)
            
            τb₁, τb₂, b_coupled = coupling_bounce(thin_1, thin_2)

            is_bounce₁ = τb₁ < τr₁
            is_bounce₂ = τb₂ < τr₂

            if(!(is_bounce₁ & is_bounce₂))
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

    return coupled_pdmp(init, kernel, coupled_kernel, dynamics, Δt, λᵣ, couple_mode)
end