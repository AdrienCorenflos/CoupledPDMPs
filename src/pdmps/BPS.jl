include("pdmp.jl")
"""
BPS kernel 
"""

function BPS_coupling(∇U::Function, H::Matrix, Δt::Float64, λᵣ::Float64, couple_mode::AbstractString = "independent")
    
    function get_thin(v::Vector, grad::Vector, H::Matrix, shift::Float64)
        
        # Set the thinning bound
        a = v'*H*v
        b = v'*grad
    
        return AffinePoisson(a, b, 0.0, shift)
    end

    # linear Dynamics
    function dynamics(x, v, t)
        return x + v*t, v
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
            return bounce(v, grad)
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
            # Progress process untill next event time > next eval time
            while(t_next < T_seq[i])
                x += v*(t_next - t)
                t = t_next
                if(is_bounce)
                    num_event += 1
                    grad = ∇U(x)
                    v = thinning_update_v(v, grad, thin, τ)
                else
                    num_event += 1
                    grad = ∇U(x)
                    v = randn(length(v))
                end
                thin = get_thin(v, grad, H, 0.0)
                τr, τb = -log(rand()) / λᵣ, rand(thin)
                τb = τb - thin.shift
                is_bounce = τb < τr
                τ = min(τr, τb)
                t_next = t + τ
            end
            # Progress event to next eval time and accumulate h
            if h_val == undef
                h_t = h(x + v*(T_seq[i] - t))
                h_val = map((u,v) -> (u + v / M), zeros(length(h_t)), h_t) #h(x + v*(T_seq[i] - t))/M
            else
                h_val = map((u,v) -> (u + v / M), h_val, h(x + v*(T_seq[i] - t))) #accumulate h
            end
        end

        thin = update_thin(thin, T_seq[n_ind] - t)
        x += v*(T_seq[n_ind] - t)
        t = T_seq[n_ind]
        new_state = (t, x, v, thin)
        return new_state, h_val, num_event
    end

    function init(x₁::Vector, x₂::Vector)

        t₁ = 0.0; t₂ = 0.0
        
        τr₁, τr₂, ref_coupled = coupling_refresh(λᵣ, Δt + t₂ - t₁, couple_mode)
        v₁, v₂, ref_pos_coupled = lindvall_roger(x₁, x₂, τr₁, τr₂) # Try and couple velocities
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

        state_1 = (t₁, x₁, v₁, thin_1)
        state_2 = (t₂, x₂, v₂, thin_2)
        
        return state_1, state_2, next_event_info, false, false
    end

    function coupled_kernel(state_1, state_2, next_event_info::coupled_event_info, coupled_next, coupled)
        t₁, x₁, v₁, thin_1 = state_1
        t₂, x₂, v₂, thin_2 = state_2

        τ₁, τ₂ = next_event_info.τ₁, next_event_info.τ₂

        if(next_event_info.bounce₁ & next_event_info.bounce₂)
            
            coupled_t = next_event_info.b_coupled
            coupled_x = false; coupled_v = false
            ref_pos_coupled = false

            t₁ += τ₁
            t₂ += τ₂

            x₁, v₁ = dynamics(x₁, v₁, τ₁)
            x₂, v₂ = dynamics(x₂, v₂, τ₂)

            grad_1, grad_2 = ∇U(x₁), ∇U(x₂)
            common_u = rand()
            # if(coupled)
            #     println("t₁:", t₁," o v₁", v₁, " v₂ ", v₂, " ", v₁ == v₂)
            # end
            v₁ = thinning_update_v(v₁, grad_1, thin_1, τ₁, common_u)
            v₂ = thinning_update_v(v₂, grad_2, thin_2, τ₂, common_u)
            # if(coupled)
            #     println("n v₁", v₁, " v₂ ", v₂, " ", v₁ == v₂)
            # end
            
            # Compute next event times. We can do this after the event given they are only bounces
            thin_1 = get_thin(v₁, grad_1, H,  0.0)
            thin_2 = get_thin(v₂, grad_2, H, Δt + t₂ - t₁)
            
            τb₁, τb₂, b_coupled = coupling_bounce(thin_1, thin_2)
            τr₁, τr₂, ref_coupled = coupling_refresh(λᵣ, Δt + t₂ - t₁, couple_mode)

            is_bounce₁ = τb₁ < τr₁
            is_bounce₂ = τb₂ < τr₂

        elseif(next_event_info.bounce₁)
            
            coupled_t = false; coupled_x = false; coupled_v = false
            ref_pos_coupled = false

            t₁ += τ₁
            t₂ += τ₂

            x₁, v₁ = dynamics(x₁, v₁, τ₁)
            x₂, v₂ = dynamics(x₂, v₂, τ₂)

            grad_1, grad_2 = ∇U(x₁), ∇U(x₂) 
            v₁ = thinning_update_v(v₁, grad_1, thin_1, τ₁)
            v₂ = randn(length(v₂))

            # Compute next event times. We can do this after the event given they are only bounces
            thin_1 = get_thin(v₁, grad_1, H,  0.0)
            thin_2 = get_thin(v₂, grad_2, H, Δt + t₂ - t₁)
            
            τb₁, τb₂, b_coupled = coupling_bounce(thin_1, thin_2)
            τr₁, τr₂, ref_coupled = coupling_refresh(λᵣ, Δt + t₂ - t₁, couple_mode)

            is_bounce₁ = τb₁ < τr₁
            is_bounce₂ = τb₂ < τr₂

        elseif(next_event_info.bounce₂)
            
            coupled_t = false; coupled_x = false; coupled_v = false
            ref_pos_coupled = false

            t₁ += τ₁
            t₂ += τ₂

            x₁, v₁ = dynamics(x₁, v₁, τ₁)
            x₂, v₂ = dynamics(x₂, v₂, τ₂)

            grad_1, grad_2 = ∇U(x₁), ∇U(x₂) 
            v₂ = thinning_update_v(v₂, grad_2, thin_2, τ₂)
            v₁ = randn(length(v₁))
            
            # Compute next event times. We can do this after the event given they are only bounces
            thin_1 = get_thin(v₁, grad_1, H,  0.0)
            thin_2 = get_thin(v₂, grad_2, H, Δt + t₂ - t₁)
            
            τb₁, τb₂, b_coupled = coupling_bounce(thin_1, thin_2)
            τr₁, τr₂, ref_coupled = coupling_refresh(λᵣ, Δt + t₂ - t₁, couple_mode)
            
            is_bounce₁ = τb₁ < τr₁
            is_bounce₂ = τb₂ < τr₂

        else
            
            coupled_t = next_event_info.ref_coupled 
            coupled_x = next_event_info.ref_pos_coupled

            t₁ += next_event_info.τ₁
            t₂ += next_event_info.τ₂

            x₁ += next_event_info.τ₁*v₁
            x₂ += next_event_info.τ₂*v₂

            # Update the refreshment PRIOR to the velocity!
            τr₁, τr₂, ref_coupled = coupling_refresh(λᵣ, Δt + t₂ - t₁, couple_mode)

            if( !(coupled_t & ref_coupled) )
                # Check if linval or modified for eff
                v₁, v₂, ref_pos_coupled = lindvall_roger(x₁, x₂, τr₁, τr₂) 
                #v₁, v₂, ref_pos_coupled = modified_lindvall_roger(x₁, x₂, τr₁, τr₂) # Try and couple velocities
            else
                # τr₁ = τr₂
                v₁, v₂, ref_pos_coupled = reflection_maximal(x₁, x₂, τr₁) 
            end
            v₁ = (v₁- x₁) ./ τr₁; v₂ = (v₂ - x₂) ./ τr₂; 

            grad_1, grad_2 = ∇U(x₁), ∇U(x₂) 
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