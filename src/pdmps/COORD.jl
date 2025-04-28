include("pdmp.jl")
"""
COORDINATE SAMPLER kernel 
"""

mutable struct COORD_coupled_status <: coupled_info
    coupled::Bool
    coupled_next::Bool
    coupled_x::BitVector
    coupled_t::Bool
    j_1::Int
    j_2::Int
    stoch_time::Float64
    num_grad_1::Int
    num_grad_2::Int
    num_grad_coupled::Int
end

struct COORD_coupled_next_event
    """ Event information for next coupled coord sampler
    Next event times, flags for if event is coupling
    """
    τ₁::Float64
    τ₂::Float64
    grad₁::Vector
    grad₂::Vector
    coupled_x::BitVector
    coupled_t::Bool
    coupled::Bool
end

function COORD_coupling(∇U::Function, H::Vector, Δt::Float64, ΔM::Int, λᵣ::Float64, h_::Function = (x) -> 0., continuous::Bool = false, couple_mode::AbstractString = "antithetic")
    
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

    function get_j(v)
        # Return which index is moving for Coordinate 0 if v = 0
        j = findfirst(abs.(v) .> 0.01)
        return j !== nothing ? j : 0
    end

    function bounce_coord(v, grad, j, u = rand())
        # Bouncing kernel for Coord sampler
        if j == 0 
            v .= 0
        else
            v .= 0
            prob_ref = λᵣ/(2*λᵣ + abs(grad[j])) 
            if u < prob_ref
                v[j] = sign(grad[j])
            else 
                v[j] = -sign(grad[j])
            end
        end
        return v
    end
    
    function dynamics(x, v, t)
        # Linear dynamics
        return x + v*t, v
    end

    function bounce_prob(v, grad, thin, τ)
        # Apply the bounce kernel with thinning (ref included in event)
        switch_rate = max(v'*grad, 0.0) + λᵣ
        upper_bound = rate(thin, τ)
        if upper_bound  < switch_rate - 1e-8
            println("upper bound", upper_bound, " ", thin)
            println("actual rate", switch_rate, " ", grad, " v ", v, " t ",τ,"\n\n")
        end
        return switch_rate/upper_bound
    end

    function get_thin_bound(j::Int, v::Vector, grad::Vector, shift::Float64, scale::Float64)
        if j == 0
            # Update time v = 0
            return AffinePoisson(0.0, λᵣ, 0.0, shift, scale)
        else
            # Update thinning bound based on a bound H for the hessian
            return AffinePoisson(H[j], grad[j]*v[j], λᵣ, shift, scale)
        end
    end

    function get_thin(z, j)
        # Get the next bounce event using thinning
        # return event time, number grads used, final gradient
        t, x, v, thin = z
        num_grad = 0
        grad = zeros(length(x))

        τb = rand(thin)
        τb = (τb - thin.shift)/thin.scale
        τ = τb
        while true

            x, v = dynamics(x, v, τb)
            grad = ∇U(x); num_grad += 1

            prb = bounce_prob(v, grad, thin, τb)
            if rand() < prb
                return τ, grad, num_grad
            end
            
            thin = get_thin_bound(j, v, grad, 0.0, 1.0)
            τb = rand(thin)
            τ += τb
        end

    end

    function kernel(state, time_seq, h_val, coupled_status = undef, next_event_info = undef, process = 1)
        # BPS kernel move state along time_seq accumulating h_val
        # Return new state, h_val and number of gradient evals

        num_grad = 0
        n_ind = length(time_seq)
        t, x, v, thin = state
        j = j0 = get_j(v)

        if !(next_event_info == undef)
            τ = process == 1 ? next_event_info.τ₁ : next_event_info.τ₂
            grad = process == 1 ? next_event_info.grad₁ : next_event_info.grad₂
        else
            # Get next time
            τ, grad, num_grad_bounce = get_thin(state, j0)
            num_grad += num_grad_bounce
        end

        if !(coupled_status == undef) & (j0 > 0)
            coupled_status.coupled_x[j0] = false
        end

        t_next = t + τ
        
        for i in 1:n_ind
            # Accumulate/move process untill next event > next time on time_seq
            while t_next < time_seq[i]
                h_val = accumulate_h(h_val, (t, x, v), time_seq, t_next)
                x, v = dynamics(x, v, t_next-t)
                t = t_next

                # Bounce coordinate sampler
                num_grad += 1
                grad = ∇U(x)
                w = abs.(grad) .+ λᵣ*2
                pushfirst!(w, λᵣ)
                w /= sum(w)
                W = Categorical(w)
                j = rand(W) - 1
                v = bounce_coord(v, grad, j)  

                if !(coupled_status == undef) & (j > 0)
                    coupled_status.coupled_x[j] = false
                end

                # Get next event time
                thin = get_thin_bound(j, v, grad, 0.0, 1.0)
                z = (t, x, v, thin)
                τ, grad, num_grad_bounce = get_thin(z, j)
                num_grad += num_grad_bounce
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
            if process == 1
                coupled_status.j_1 = j
                coupled_status.num_grad_1 += num_grad
            else
                coupled_status.j_2 = j
                coupled_status.num_grad_2 += num_grad
            end
        end

        # Update linear thinning to be valid at final time
        new_state = (t, x, v, thin)
        # print(coupled_status.j_1, coupled_status.j_2)
        return new_state, h_val, coupled_status
    end

    function coupling_bounce(z₁, z₂, coupled_index, j_1, j_2)

        t₁, x₁, v₁, thin_1 = z₁
        t₂, x₂, v₂, thin_2 = z₂
        if coupled_index & (j_1 > 0)
            # After Kernel possible index may be coupled so couple position
            thin_1 = AffinePoisson(thin_1.a, thin_1.b, thin_1.c, x₁[j_1], v₁[j_1])
            thin_2 = AffinePoisson(thin_2.a, thin_2.b, thin_2.c, x₂[j_1], v₂[j_1])
        end
        num_grad_1, num_grad_2 = 0, 0
        grad₁, grad₂ = zeros(length(x₁)), zeros(length(x₁))

        if j_1 == 0 && coupled_index
            ## Couple time not position (Exact simulation possible)
            τ₁, τ₂, coupled_event = coupling_refresh(λᵣ, Δt + t₂ - t₁, couple_mode)
            return τ₁, τ₂, coupled_event, num_grad_1, num_grad_2, ∇U(x₁), ∇U(x₂) # Should carry these rather than recalculate
        else
            ## Get next times using thinning.
            τ₁, τ₂ = 0.0, 0.0
            test=0
            while true
                test += 1
                τb₁, τb₂, coupled_event = thorisson(thin_1, thin_2)
                τb₁ = (τb₁ - thin_1.shift)/thin_1.scale
                τb₂ = (τb₂ - thin_2.shift)/thin_2.scale

                x₁, v₁ = dynamics(x₁, v₁, τb₁)
                τ₁ = τ₁ + τb₁
                grad₁ = ∇U(x₁)
                num_grad_1 += 1
                a₁ = bounce_prob(v₁, grad₁, thin_1, τb₁)

                x₂, v₂ = dynamics(x₂, v₂, τb₂)
                τ₂ = τ₂ + τb₂
                grad₂ = ∇U(x₂)
                num_grad_2 += 1
                a₂ = bounce_prob(v₂, grad₂, thin_2, τb₂)
                u = rand()

                if u < min(a₁, a₂) 
                    return τ₁, τ₂, coupled_event, num_grad_1, num_grad_2, grad₁, grad₂
                elseif u < a₁
                    # Continue thinning process 2 
                    thin_2 = AffinePoisson(H[j_2], grad₂[j_2]*v₂[j_2], λᵣ)
                    z = (t₂ + τ₂, x₂, v₂, thin_2)
                    τb, grad₂, num_grad_bounce = get_thin(z, j_2)
                    τ₂ += τb; num_grad_2 += num_grad_bounce
                    return τ₁, τ₂, false, num_grad_1, num_grad_2, grad₁, grad₂
                elseif u < a₂
                    # Continue thinning process 1
                    thin_1 = AffinePoisson(H[j_1], grad₁[j_1]*v₁[j_1], λᵣ)
                    z = (t₁+ τ₁, x₁, v₁, thin_1)
                    τb, grad₁, num_grad_bounce = get_thin(z, j_1)
                    τ₁ += τb; num_grad_1 += num_grad_bounce
                    return τ₁, τ₂, false, num_grad_1, num_grad_2, grad₁, grad₂
                end

                # If u > a₁ and a₂ then continue coupled thinning
                if coupled_index & (j_1 > 0)
                    # Couple position of j'th coord
                    thin_1 = AffinePoisson(H[j_1], grad₁[j_1]*v₁[j_1], λᵣ, x₁[j_1], v₁[j_1])
                    thin_2 = AffinePoisson(H[j_2], grad₂[j_2]*v₂[j_2], λᵣ, x₂[j_2], v₂[j_2])
                else
                    # Couple time
                    thin_1 = get_thin_bound(j_1, v₁, grad₁, 0.0, 1.0)
                    thin_2 = get_thin_bound(j_2, v₂, grad₂, Δt + t₂ - t₁, 1.0)
                end
            end 
        end
    end

    function get_next_event(state_1, state_2, coupled_status)
        t₁, x₁, v₁, thin_1 = state_1
        t₂, x₂, v₂, thin_2 = state_2
        
        j_1, j_2 = coupled_status.j_1, coupled_status.j_2
        coupled_index = (j_1 == j_2)

        τ₁, τ₂, coupled_event, num_grad_1, num_grad_2, grad₁, grad₂ = coupling_bounce(state_1, state_2, coupled_index, j_1, j_2)

        # Update the count on events
        if coupled_status.coupled
            coupled_status.num_grad_coupled += num_grad_1
        else
            coupled_status.num_grad_1 += num_grad_1
            coupled_status.num_grad_2 += num_grad_2
        end

        # Determine next event coupled status
        coupled_t = coupled_status.coupled_t
        coupled_x = coupled_status.coupled_x
        coupled = coupled_status.coupled

        if !coupled_status.coupled
            if coupled_index
                if j_1 == 0
                    coupled_t = coupled_event
                    coupled = (all(coupled_x) & coupled_event)
                else
                    coupled_t = false
                    coupled_x[j_1] = coupled_event

                end
            else
                # If bounce is not coupled (j₁\neq j₂) the process may still couple in time but will be uncoupled in position
                coupled_t = coupled_event
                if j_1 > 0
                    coupled_x[j_1] = false
                end
                if j_2 > 0 
                    coupled_x[j_2] = false
                end
            end
        end
        
        next_event = COORD_coupled_next_event(τ₁, τ₂, grad₁, grad₂, coupled_x, coupled_t, coupled)
        return next_event, coupled_status
    end

    function init(x₁::Vector, x₂::Vector)
        # COORD Init coupled sampler 
        # Progress Z1 by Δ

        t₁ = 0.0; t₂ = 0.0
        
        # initialise with random velocity
        d = length(x₁)
        v₁, v₂ = zeros(d), zeros(d)
        j_1, j_2 = rand(1:d), rand(1:d)
        v₁[j_1], v₂[j_2] = rand([1,-1]), rand([1,-1])

        # Get thinning for next event
        grad₁, grad₂ = ∇U(x₁), ∇U(x₂)

        thin_1 = get_thin_bound(j_1, v₁, grad₁, 0.0, 1.0)
        thin_2 = get_thin_bound(j_2, v₂, grad₂, Δt + t₂ - t₁, 1.0)

        state_1 = (t₁, x₁, v₁, thin_1)
        state_2 = (t₂, x₂, v₂, thin_2)

        # Get next event times and set couple status (assume not coupled)
        coupled_status = COORD_coupled_status(false, false, falses(d), false, j_1, j_2, Inf, 0, 0, 0)
        next_event_info, coupled_status = get_next_event(state_1, state_2, coupled_status)

        return state_1, state_2, next_event_info, coupled_status, false
    end

    function coupled_kernel(state_1, state_2, next_event_info::COORD_coupled_next_event, coupled_status::COORD_coupled_status, coupled)
        
        t₁, x₁, v₁, thin_1 = state_1
        t₂, x₂, v₂, thin_2 = state_2

        τ₁, τ₂ = next_event_info.τ₁, next_event_info.τ₂

        # Update States
        t₁ += τ₁
        t₂ += τ₂

        x₁, v₁ = dynamics(x₁, v₁, τ₁)
        x₂, v₂ = dynamics(x₂, v₂, τ₂)
        
        # Update coupling status 

        if next_event_info.coupled & !coupled_status.coupled
            coupled = coupled_status.coupled = next_event_info.coupled
            coupled_status.stoch_time = t₁
        end
        coupled_status.coupled_x = next_event_info.coupled_x
        coupled_status.coupled_t = next_event_info.coupled_t

        ### Bounce kernel 

        ## First choose the index to couple (1,2,..., d)
        # index 1 (v=0) has probability λ, index i has prob = |grad(i)| + 2λ 
        w1, w2 = abs.(next_event_info.grad₁) .+ λᵣ*2, abs.(next_event_info.grad₂) .+ λᵣ*2
        pushfirst!(w1, λᵣ); pushfirst!(w2, λᵣ)
        w1 /= sum(w1); w2 /= sum(w2)
        W1, W2 = Categorical(w1), Categorical(w2)

        j_1, j_2, coupled_index = thorisson(W1,W2)
        j_1, j_2 = j_1 - 1, j_2 - 1

        coupled_status.j_1 = j_1
        coupled_status.j_2 = j_2

        common_u = rand()
        v₁ = bounce_coord(v₁, next_event_info.grad₁, j_1, common_u)
        v₂ = bounce_coord(v₂, next_event_info.grad₂, j_2, common_u)

        ## Set thinning bounds 
        if coupled_index & (j_1 > 0)
            # Couple position of j'th coord
            thin_1 = AffinePoisson(H[j_1], next_event_info.grad₁[j_1]*v₁[j_1], λᵣ, x₁[j_1], v₁[j_1])
            thin_2 = AffinePoisson(H[j_2], next_event_info.grad₂[j_2]*v₂[j_2], λᵣ, x₂[j_2], v₂[j_2])
        else
            # Couple time
            thin_1 = get_thin_bound(j_1, v₁, next_event_info.grad₁, 0.0, 1.0)
            thin_2 = get_thin_bound(j_2, v₂, next_event_info.grad₂, Δt + t₂ - t₁, 1.0)
        end

        state_1 = (t₁, x₁, v₁, thin_1)
        state_2 = (t₂, x₂, v₂, thin_2)
        next_event_info, coupled_status = get_next_event(state_1, state_2, coupled_status)
        
        if !coupled
            coupled_status.num_grad_1 += 1
            coupled_status.num_grad_2 += 1
        else
            coupled_status.num_grad_coupled += 1
        end

        return state_1, state_2, next_event_info, coupled_status, coupled
    end

    return coupled_pdmp(init, kernel, coupled_kernel, dynamics, λᵣ, Δt, ΔM, accumulate_h, get_next_event)
end