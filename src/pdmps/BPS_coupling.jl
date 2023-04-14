include("BPS.jl")
include("../generic_couplings/dau_chopin.jl")
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
    coupled_x::Bool
    coupled::Bool
end

function crn(d₁, d₂)
    function Γ(rng=Random.GLOBAL_RNG)
        copied_rng = copy(rng)
        x₁ = rand(rng, d₁)
        x₂ = rand(copied_rng, d₂)
        return x₁, x₂
    end
    return Γ
end

function rand(state_1::BPSstate, state_2::BPSstate)
    """ Make a coupled proposal
    Return next time and if refreshment event
    """
    # couple refreshment
    Γᵣ = crn(state_1.refresh, state_2.refresh)
    ref1, ref2, coupled_time_ref = dau_chopin(state_1.refresh, state_2.refresh, Γᵣ)
    Γₜ = crn(state_1.thinning, state_2.thinning)
    thin1, thin2, coupled_time_thin = dau_chopin(state_1.thinning, state_2.thinning, Γₜ)
    
    τ1, τ2 = min(ref1, thin1), min(ref2, thin2)
    refresh1, refresh2 = ref1 < thin1, ref2 < thin2
    if(refresh1)
        coupled_time = refresh2 & coupled_time_ref
    else
        coupled_time = !refresh2 & coupled_time_thin
    end

    τ2 = τ2 - state_2.thinning.shift

    return coupled_time, refresh1, refresh2, τ1, τ2
end


function BPS_coupling(∇U::Function, H::Matrix, h::Function, Δt::Float64, λᵣ::Float64)

    function kernel_event(state_1::BPSstate, state_2::BPSstate, coupled_x::Bool, coupled::Bool)
        
        coupled_time, refresh1, refresh2, τ1, τ2 = rand(state_1, state_2)

        t1, x1, v1 = move_linear(state_1.skeleton, τ1)
        t2, x2, v2 = move_linear(state_2.skeleton, τ2)
      
        rng = Random.default_rng()
        current_rng = copy(rng)

        if( coupled )
            a1, b1, v1, _ = bounce_kernel(rng, state_1, H, ∇U(x1), refresh1, τ1)
            a2, b2, v2 = copy(a1), copy(b1), copy(v1)
        else
            if(refresh1 & refresh2)
                Γᵣ = crn(HomogeneousPoisson(λᵣ), HomogeneousPoisson(λᵣ, t2+Δt-t1))
                τ1_next, _, coupled_time_next = dau_chopin(HomogeneousPoisson(λᵣ), HomogeneousPoisson(λᵣ, t2+Δt-t1), Γᵣ)
                v1, v2, coupled_v = reflection_maximal(x1, x2, τ1_next)
                v1 -= x1; v2 -= x2; 
                v1 /= τ1_next; v2 /= τ1_next; 
                
                if(coupled_x)
                    coupled = coupled_time & coupled_v
                else
                    coupled_x = coupled_time_next & coupled_v & coupled_time
                end 
                copy!(Random.default_rng(), current_rng)
                a1 = v1'*H*v1; a2 = v2'*H*v2;
                b1 = v1'*∇U(x1); b2 = v2'*∇U(x2) 

            else
                coupled_x = false
                a1, b1, v1, _ = bounce_kernel(rng, state_1, H, ∇U(x1), refresh1, τ1)
                a2, b2, v2, _ = bounce_kernel(current_rng, state_2, H, ∇U(x2), refresh2, τ2)
            end
        end


        newskeleton1 = Skeleton(t1, x1, v1)
        newstate_1 = BPSstate(newskeleton1, AffinePoisson(a1,b1), HomogeneousPoisson(λᵣ))
        
        newskeleton2 = Skeleton(t2, x2, v2)
        newstate_2 = BPSstate(newskeleton2, AffinePoisson(a2,b2,0., t2+Δt-t1), HomogeneousPoisson(λᵣ, t2+Δt-t1))

        return newstate_1, newstate_2, coupled_x, coupled
    end

    function kernel(coupledstate::BPScoupledstate)
        current_1, current_2 = coupledstate.state_1.current, coupledstate.state_2.current
        event_vec_1, event_vec_2 = coupledstate.state_1.event_vec, coupledstate.state_2.event_vec

        coupled_x, coupled = coupledstate.coupled_x, coupledstate.coupled

        update_event_1 = event_vec_1[end].skeleton.t < current_1.t + Δt 
        update_event_2 = event_vec_2[end].skeleton.t < current_2.t + Δt 
        
        # Update event list until it contains next events for both processes
        while(update_event_1 || update_event_2)

            event_1, event_2, coupled_x, coupled = kernel_event(event_vec_1[end], event_vec_2[end], coupled_x, coupled)
            event_vec_1 = vcat(event_vec_1, event_1)
            event_vec_2 = vcat(event_vec_2, event_2)

            update_event_1 = event_1.skeleton.t < current_1.t + Δt 
            update_event_2 = event_2.skeleton.t < current_2.t + Δt 
        end

        new_current_1 = move(current_1, event_vec_1, Δt)
        new_current_2 = move(current_2, event_vec_2, Δt)

        newstate_1 = BPSDiscreteState(new_current_1, event_vec_1, h(new_current_1))
        newstate_2 = BPSDiscreteState(new_current_2, event_vec_2, h(new_current_2))
        #coupled = (sum(abs.(new_current_1.x .- new_current_2.x)) < 1e-10) & (sum(abs.(new_current_1.v .- new_current_2.v)) < 1e-10) & (new_current_1.t - Δt - new_current_1.t  < 1e-10)
        newstate = BPScoupledstate(newstate_1, newstate_2, coupled_x, coupled)
        return newstate
    end

    function init(position::Vector, velocity::Vector)
        grad = ∇U(position)

        # Set the thinning bound
        a = velocity'*H*velocity
        b = velocity'*grad

        newskeleton = Skeleton(0., position, velocity)
        return BPSstate(newskeleton, AffinePoisson(a,b), HomogeneousPoisson(λᵣ))
    end

    function init_coupling(position1::Vector, velocity1::Vector, position2::Vector, velocity2::Vector)
        event_1 = init(position1, velocity1)
        new_current_1 = event_1.skeleton

        temp = BPSDiscreteState(new_current_1, [event_1], h(new_current_1))

        nextstate = kernel(BPScoupledstate(temp, temp, false, false))
        state_1 = BPSDiscreteState(nextstate.state_1.current, [nextstate.state_1.event_vec[1]], h(nextstate.state_1.current))

        event_2 = init(position2, velocity2)
        new_current_2 = event_2.skeleton
        state_2 = BPSDiscreteState(new_current_2, [event_2], h(new_current_2))
        
        return BPScoupledstate(state_1, state_2, false, false)
    end

    return DPDMP(init_coupling, kernel, kernel_event)
end

