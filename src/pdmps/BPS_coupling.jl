include("BPS.jl")

"""
State of the Bouncy Particle Sampler

Additional information regarding the thinning procedure is stored for efficiency.

# Fields
- `skeleton::Skeleton`: The skeleton of the PDMP
- `thinning::AffinePoisson`: The proposal for thinning 
"""

struct BPScoupledstate <: PDMPState
    """ State of the BPS
    The BPS algo takes a position and returns the next thinned event.
    To improve computation on the bounce and thinning additional info is stored
    """
    state_1::BPSDiscreteState
    state_2::BPSDiscreteState
    coupled::Bool
end

function rand(state_1::BPSstate, state_2::BPSstate)
    """ Make a coupled proposal
    Return next time and if refreshment event
    """
    λᵣ = state_1.thinning.c                         # Assumes the same ref rate for both

    τ1, τ2, coupled_time = thorisson(state_1.thinning, state_2.thinning)
    τ2 = τ2 - state_2.thinning.shift
    λ1, λ2 = rate(state_1.thinning, τ1), rate(state_2.thinning, τ2)

    refresh1, refresh2, coupled = thorisson(Bernoulli(λᵣ/λ1), Bernoulli(λᵣ/λ2)) # Lazy coupling

    return coupled_time, refresh1, refresh2, τ1, τ2
end


function BPS_coupling(∇U::Function, H::Matrix, h::Function, Δt::Float64, λᵣ::Float64)

    function coupled_kernel_event(state_1::BPSstate, state_2::BPSstate)
        
        coupled_time, refresh1, refresh2, τ1, τ2 = rand(state_1, state_2)

        t1, x1, v1 = move_linear(state_1.skeleton, τ1)
        t2, x2, v2 = move_linear(state_2.skeleton, τ2)
      
        rng = Random.default_rng()
        current_rng = copy(rng)

        if(refresh1 & refresh2)
            if(coupled_time)
                v1, v2, _, _, _ = coupled_next_refresh(x1, x2, λᵣ)
                copy!(Random.default_rng(), current_rng)
            else
                v1, v2, _ = reflection_maximal(fill(0.,size(v1)), fill(0.,size(v1)), 1.)
            end
            a1 = v1'*H*v1; a2 = v2'*H*v2;
            b1 = v1'*∇U(x1); b2 = v2'*∇U(x2) 
            event1 = true; event2 = true
        else
            a1, b1, v1, event1 = bounce_kernel(rng, state_1, H, ∇U(x1), refresh1, τ1)
            a2, b2, v2, event2 = bounce_kernel(current_rng,state_2, H, ∇U(x2), refresh2, τ2)
        end

        newskeleton1 = Skeleton(t1, x1, v1)
        newstate_1 = BPSstate(newskeleton1, AffinePoisson(a1,b1,λᵣ))
        
        newskeleton2 = Skeleton(t2, x2, v2)
        newstate_2 = BPSstate(newskeleton2, AffinePoisson(a2,b2,λᵣ, t2+Δt-t1))

        return newstate_1, newstate_2
    end

    function kernel(coupledstate::BPScoupledstate)
        current_1, current_2 = coupledstate.state_1.current, coupledstate.state_2.current
        event_vec_1, event_vec_2 = coupledstate.state_1.event_vec, coupledstate.state_2.event_vec

        update_event_1 = event_vec_1[end].skeleton.t < current_1.t + Δt 
        update_event_2 = event_vec_2[end].skeleton.t < current_2.t + Δt 
        
        # Update event list until it contains next events for both processes
        while(update_event_1 || update_event_2)

            event_1, event_2 = coupled_kernel_event(event_vec_1[end], event_vec_2[end])
            event_vec_1 = vcat(event_vec_1, event_1)
            event_vec_2 = vcat(event_vec_2, event_2)

            update_event_1 = event_1.skeleton.t < current_1.t + Δt 
            update_event_2 = event_2.skeleton.t < current_2.t + Δt 
        end

        new_current_1 = move(current_1, event_vec_1, Δt)
        new_current_2 = move(current_2, event_vec_2, Δt)

        newstate_1 = BPSDiscreteState(new_current_1, event_vec_1, h(new_current_1))
        newstate_2 = BPSDiscreteState(new_current_2, event_vec_2, h(new_current_2))
        coupled = (sum(abs.(new_current_1.x .- new_current_2.x)) < 1e-10) & (sum(abs.(new_current_1.v .- new_current_2.v)) < 1e-10) & (new_current_1.t - Δt - new_current_1.t  < 1e-10)
        newstate = BPScoupledstate(newstate_1, newstate_2, coupled)
        return newstate
    end

    function init(position::Vector)
        velocity = randn(length(position))
        grad = ∇U(position)

        # Set the thinning bound
        a = velocity'*H*velocity
        b = velocity'*grad

        newskeleton = Skeleton(0., position, velocity)
        return BPSstate(newskeleton, AffinePoisson(a,b,λᵣ))
    end

    function init_coupling(position1::Vector, position2::Vector)
        event_1 = init(position1)
        new_current_1 = event_1.skeleton

        temp = BPSDiscreteState(new_current_1, [event_1], h(new_current_1))

        nextstate = kernel(BPScoupledstate(temp, temp, false))
        state_1 = BPSDiscreteState(nextstate.state_1.current, [nextstate.state_1.event_vec[1]], h(nextstate.state_1.current))

        event_2 = init(position2)
        new_current_2 = event_2.skeleton
        state_2 = BPSDiscreteState(new_current_2, [event_2], h(new_current_2))
        
        return BPScoupledstate(state_1, state_2, false)
    end

    return PDMP(init_coupling, kernel)
end

## The functions/structs below are temporary and return the events rather than the descrete time kernel
## This version will run correctly and can be used for testing.
struct BPSalt <: PDMPState
    """ bl
    The BPS algo takes a position and returns the next thinned event.
    """
    state_1::BPSstate
    state_2::BPSstate
    coupled::Bool
end

function BPS_coupling2(∇U::Function, H::Matrix, h::Function, Δt::Float64, λᵣ::Float64)

    function coupled_kernel_event(state::BPSalt)
        state_1 = state.state_1
        state_2 = state.state_2

        coupled_time, refresh1, refresh2, τ1, τ2 = rand(state_1, state_2)

        t1, x1, v1 = move_linear(state_1.skeleton, τ1)
        t2, x2, v2 = move_linear(state_2.skeleton, τ2)
      
        rng = Random.default_rng()
        current_rng = copy(rng)

        if(refresh1 & refresh2)
            if(coupled_time)
                coupled_next_refresh!(x1, x2, λᵣ, v1, v2)
                copy!(Random.default_rng(), current_rng)
            else
                reflection_maximal!(fill(0.,size(v1)), fill(0.,size(v1)), 1., v1, v2)
            end
            a1 = v1'*H*v1; a2 = v2'*H*v2;
            b1 = v1'*∇U(x1); b2 = v2'*∇U(x2) 
            event1 = true; event2 = true
        else
            a1, b1, v1, event1 = bounce_kernel(rng, state_1, H, ∇U(x1), refresh1, τ1)
            a2, b2, v2, event2 = bounce_kernel(current_rng,state_2, H, ∇U(x2), refresh2, τ2)
        end

        newskeleton1 = Skeleton(t1, x1, v1)
        newstate_1 = BPSstate(newskeleton1, AffinePoisson(a1,b1,λᵣ))
        
        newskeleton2 = Skeleton(t2, x2, v2)
        newstate_2 = BPSstate(newskeleton2, AffinePoisson(a2,b2,λᵣ, t2+Δt-t1))

        return BPSalt(newstate_1, newstate_2, false)
    end
    function kernel(coupledstate::BPScoupledstate)
        current_1, current_2 = coupledstate.state_1.current, coupledstate.state_2.current
        event_vec_1, event_vec_2 = coupledstate.state_1.event_vec, coupledstate.state_2.event_vec

        update_event_1 = event_vec_1[end].skeleton.t < current_1.t + Δt 
        update_event_2 = event_vec_2[end].skeleton.t < current_2.t + Δt 

        while(update_event_1 || update_event_2)

            tmp2 = BPSalt(event_vec_1[end], event_vec_2[end], false)
            tmp = coupled_kernel_event(tmp2)
            event_1 = tmp.state_1
            event_2 = tmp.state_2
            push!(event_vec_1, event_1)
            push!(event_vec_2, event_2)

            update_event_1 = event_vec_1[end].skeleton.t < current_1.t + Δt 
            update_event_2 = event_vec_2[end].skeleton.t < current_2.t + Δt 
        end

        new_current_1 = move(current_1, event_vec_1, Δt)
        new_current_2 = move(current_2, event_vec_2, Δt)

        newstate_1 = BPSDiscreteState(new_current_1, event_vec_1, h(new_current_1))
        newstate_2 = BPSDiscreteState(new_current_2, event_vec_2, h(new_current_2))
        coupled = (sum(abs.(new_current_1.x .- new_current_2.x)) < 1e-10) & (sum(abs.(new_current_1.v .- new_current_2.v)) < 1e-10) & (new_current_1.t - Δt - new_current_1.t  < 1e-10)
        newstate = BPScoupledstate(newstate_1, newstate_2, coupled)
        return newstate
    end
    function init(position::Vector)
        velocity = randn(length(position))
        grad = ∇U(position)

        # Set the thinning bound
        a = velocity'*H*velocity
        b = velocity'*grad

        newskeleton = Skeleton(0., position, velocity)
        return BPSstate(newskeleton, AffinePoisson(a,b,λᵣ))
    end

    function init_coupling(position1::Vector, position2::Vector)
        event_1 = init(position1)
        new_current_1 = event_1.skeleton

        temp = BPSDiscreteState(new_current_1, [event_1], h(new_current_1))

        nextstate = kernel(BPScoupledstate(temp, temp, false))
        state_1 = BPSDiscreteState(nextstate.state_1.current, [nextstate.state_1.event_vec[1]], h(nextstate.state_1.current))

        event_2 = init(position2)
        new_current_2 = event_2.skeleton
        state_2 = BPSDiscreteState(new_current_2, [event_2], h(new_current_2))
        
        return BPSalt(nextstate.state_1.event_vec[1], event_2, false)
    end

    return PDMP(init_coupling, coupled_kernel_event)
end
