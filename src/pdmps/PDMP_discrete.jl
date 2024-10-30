include("BPS.jl")
include("BOOM.jl")

struct MyState 
    z::Any
    h::Any
end

mutable struct coupled_info
    coupled_next::Bool
    stoch_time::Float64
    num_events_1::Int
    num_events_2::Int
    num_event_coupled::Int
end

struct MyCoupledState
    state_1::MyState
    state_2::MyState
    coupled_status::coupled_info
    coupled::Bool
end

# Set the thinning bound
function update_thin(thin::AffinePoisson, t)
    a = thin.a
    b = thin.a*t + thin.b # Assumes an affine bound
    shift = 0.0 # Assumes the updated thinning is time-coupled
    return AffinePoisson(a, b, 0.0, shift)
end

function DiscretePDMPCoupling(sampler::coupled_pdmp)

    pdmp_kernel = sampler.kernel
    coupled_pdmp_kernel = sampler.coupled_kernel
    dynamics = sampler.dynamics

    λᵣ = sampler.λᵣ
    Δt = sampler.Δt
    ΔM = sampler.ΔM
    
    δ = Δt/ΔM

    h = sampler.h
    accumulate_h = sampler.accumulate_h
    couple_mode = sampler.couple_mode

    function get_next_event_info(t₁, t₂, thin_1, thin_2)
        τr₁, τr₂, ref_coupled = coupling_refresh(λᵣ, Δt + t₂ - t₁, couple_mode)
        τb₁, τb₂, b_coupled = coupling_bounce(thin_1, thin_2)
        is_bounce₁ = τb₁ < τr₁
        is_bounce₂ = τb₂ < τr₂
        τ₁ = min(τr₁, τb₁)
        τ₂ = min(τr₂, τb₂)
        return coupled_event_info(τ₁, τ₂, is_bounce₁, is_bounce₂, ref_coupled, false, b_coupled)
    end

    # Init
    function coupled_init(x₁::Vector, x₂::Vector)

        coupled_event = sampler.init(x₁, x₂)
        state_1, state_2, _, _, _ = coupled_event
        init_h = 0 .* h(state_1[2], state_1[3], 0)
        state_1, new_h₁, n_event_1 = pdmp_kernel(state_1, δ:δ:Δt, init_h)
        
        new_state_1 = MyState(state_1, new_h₁)
        new_state_2 = MyState(state_2, init_h)
    
        # Get next event info
        coupled_status = coupled_info(false, Inf, n_event_1, 0, 0)
        new_coupled_state = MyCoupledState(new_state_1, new_state_2, coupled_status, false)
        return new_coupled_state
    end

    function coupled_kernel(coupled_state::MyCoupledState)
        
        state_1, state_2 = coupled_state.state_1, coupled_state.state_2
        coupled_status = coupled_state.coupled_status
        coupled = coupled_state.coupled
        t₁, x₁, v₁, thin_1 = state_1.z
        t₂, x₂, v₂, thin_2 = state_2.z
    
        next_t₁, next_t₂ = t₁ + Δt, t₂ + Δt
        seq_1, seq_2 = (t₁+δ):δ:next_t₁, (t₂+δ):δ:next_t₂
    
        # Refresh the event independent of the past
        next_event_info = get_next_event_info(t₁, t₂, thin_1, thin_2)
        coupled_status.coupled_next = false
        
        #Check if next event is past the next time
        update_1 = t₁ + next_event_info.τ₁ < next_t₁
        update_2 = t₂ + next_event_info.τ₂ < next_t₂
    
        # Reset functions to accumulate over Δ
        h_1 = state_1.h .* 0
        h_2 = state_2.h .* 0

        state_1 = (t₁, x₁, v₁, thin_1)
        state_2 = (t₂, x₂, v₂, thin_2)
        
        while update_1 && update_2
            
            coupled_event = coupled_pdmp_kernel(state_1, state_2, next_event_info, coupled_status.coupled_next, coupled)
            state_1, state_2, next_event_info, coupled_status.coupled_next, coupled = coupled_event
            
            # Accumulate h over the interval
            h_1 = accumulate_h(h_1, x₁, v₁, t₁, seq_1, state_1[1])
            h_2 = accumulate_h(h_2, x₂, v₂, t₂, seq_2, state_2[1])

            t₁, x₁, v₁, thin_1 = state_1
            t₂, x₂, v₂, thin_2 = state_2
    
            update_1 = t₁ + next_event_info.τ₁ < next_t₁
            update_2 = t₂ + next_event_info.τ₂ < next_t₂
    
            if(!coupled)
                coupled_status.stoch_time = t₂
                coupled_status.num_events_1 += 1
                coupled_status.num_events_2 += 1
            else
                coupled_status.num_event_coupled += 1
            end
        end
        
        if update_1
            # Update process 1 until stoch time t_next
            ind_1 = findfirst( seq_1 .>= t₁)
            (t₁, x₁, v₁, thin_1), h_1, num_event_1 = pdmp_kernel(state_1, seq_1[ind_1:end], h_1)
            coupled_status.num_events_1 += num_event_1

            h_2 = accumulate_h(h_2, x₂, v₂, t₂, seq_2, seq_2[end])
            x₂, v₂ = dynamics(x₂, v₂, next_t₂ - t₂)
            thin_2 = update_thin(thin_2, next_t₂ - t₂)
            t₂ = next_t₂
    
        elseif update_2
            # Update process 2 to stoch time t_next
            ind_2 = findfirst( seq_2 .>= t₂)
            (t₂, x₂, v₂, thin_2), h_2, num_event_2 = pdmp_kernel(state_2, seq_2[ind_2:end], h_2)
            coupled_status.num_events_2 += num_event_2
            
            h_1 = accumulate_h(h_1, x₁, v₁, t₁, seq_1, seq_1[end])
            x₁, v₁ = dynamics(x₁, v₁, next_t₁ - t₁)
            thin_1 = update_thin(thin_1, next_t₁ - t₁)
            t₁ = next_t₁
    
        else
            # If both processess can continue moving adjust the event information and continue.
            
            h_1 = accumulate_h(h_1, x₁, v₁, t₁, seq_1, seq_1[end])
            h_2 = accumulate_h(h_2, x₂, v₂, t₂, seq_2, seq_2[end])
            x₁, v₁ = dynamics(x₁, v₁, next_t₁ - t₁)
            x₂, v₂ = dynamics(x₂, v₂, next_t₂ - t₂)
    
            thin_1 = update_thin(thin_1, next_t₁ - t₁)
            thin_2 = update_thin(thin_2, next_t₂ - t₂)
    
            t₁ = next_t₁
            t₂ = next_t₂
        end
    
        new_state_1 = MyState((t₁, x₁, v₁, thin_1), h_1)
        new_state_2 = MyState((t₂, x₂, v₂, thin_2), h_2) 
        new_coupled_state = MyCoupledState(new_state_1, new_state_2, coupled_status, coupled)
        return new_coupled_state
    end
    return Kernel(coupled_init, coupled_kernel)
end
