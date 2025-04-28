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

    accumulate_h = sampler.accumulate_h
    get_next_event = sampler.get_next_event

    # Init
    function coupled_init(x₁::Vector, x₂::Vector)

        coupled_event = sampler.init(x₁, x₂)
        state_1, state_2, _, coupled_status, _ = coupled_event

        # Evolve process 1 Δt and update number of grad evals
        state_1, new_h₁, coupled_status = pdmp_kernel(state_1, δ:δ:Δt, undef, coupled_status)
        coupled_status.coupled_t = true
        
        new_state_1 = MyState(state_1, new_h₁)
        new_state_2 = MyState(state_2, new_h₁ .* 0.0)
    
        new_coupled_state = MyCoupledState(new_state_1, new_state_2, coupled_status, false)
        return new_coupled_state
    end

    function coupled_kernel(coupled_state::MyCoupledState)
        
        state_1, state_2 = coupled_state.state_1, coupled_state.state_2
        coupled_status = coupled_state.coupled_status
        coupled = coupled_state.coupled
        
        t₁, _ = state_1.z
        t₂, _ = state_2.z
    
        next_t₁, next_t₂ = t₁ + Δt, t₂ + Δt
        seq_1, seq_2 = (t₁+δ):δ:next_t₁, (t₂+δ):δ:next_t₂
    
        # Get the next event info independent of the past
        next_event_info, coupled_status = get_next_event(state_1.z, state_2.z, coupled_status)
        
        #Check if next event is past the next time
        update_1 = t₁ + next_event_info.τ₁ < next_t₁
        update_2 = t₂ + next_event_info.τ₂ < next_t₂
    
        # Reset functions to accumulate over Δ
        h_1 = state_1.h .* 0
        h_2 = state_2.h .* 0

        z_1 = state_1.z
        z_2 = state_2.z
        
        while update_1 && update_2
            
            coupled_event = coupled_pdmp_kernel(z_1, z_2, next_event_info, coupled_status, coupled)
            z_1_new, z_2_new, next_event_info, coupled_status, coupled = coupled_event

            t₁, t₂ = z_1_new[1], z_2_new[1]
            
            # Accumulate h over the interval
            h_1 = accumulate_h(h_1, z_1, seq_1, t₁)
            h_2 = accumulate_h(h_2, z_2, seq_2, t₂)

            z_1 = z_1_new
            z_2 = z_2_new
    
            update_1 = t₁ + next_event_info.τ₁ < next_t₁
            update_2 = t₂ + next_event_info.τ₂ < next_t₂
        end

        # Progress processes so coupled in time
        # If update_1 then only update process 1, update_2 then only update process 2
        # Otherwise accumulate both processes to the next Δ with no events 

        if update_1
            # Update process 1 using Kernel
            ind_1 = findfirst( seq_1 .>= t₁ )
            z_1, h_1, coupled_status = pdmp_kernel(z_1, seq_1[ind_1:end], h_1, coupled_status, next_event_info, 1)
        else
            # progress to next_t (no events)
            h_1 = accumulate_h(h_1, z_1, seq_1, seq_1[end])
            t₁, x₁, v₁, thin_1 = z_1
            x₁, v₁ = dynamics(x₁, v₁, next_t₁ - t₁)
            thin_1 = update_thin(thin_1, next_t₁ - t₁)
            t₁ = next_t₁
            z_1 = (t₁, x₁, v₁, thin_1)
        end

        if update_2
            ind_2 = findfirst( seq_2 .>= t₂ )
            z_2, h_2, coupled_status = pdmp_kernel(z_2, seq_2[ind_2:end], h_2, coupled_status, next_event_info, 2)
        else
            h_2 = accumulate_h(h_2, z_2, seq_2, seq_2[end])
            t₂, x₂, v₂, thin_2 = z_2
            x₂, v₂ = dynamics(x₂, v₂, next_t₂ - t₂)
            thin_2 = update_thin(thin_2, next_t₂ - t₂)
            t₂ = next_t₂
            z_2 = (t₂, x₂, v₂, thin_2)
        end
        
        coupled_status.coupled_t = true
        new_state_1 = MyState(z_1, h_1)
        new_state_2 = MyState(z_2, h_2) 
        new_coupled_state = MyCoupledState(new_state_1, new_state_2, coupled_status, coupled)
        return new_coupled_state
    end
    return Kernel(coupled_init, coupled_kernel)
end
