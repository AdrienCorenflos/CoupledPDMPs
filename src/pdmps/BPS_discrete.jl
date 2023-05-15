include("BPS_coupling.jl")

struct MyState 
    z::Any
    h::Any
end

mutable struct couple_info
    stoch_time::Float64
    num_events_1::Int
    num_events_2::Int
end

struct MyCoupledState
    state_1::MyState
    state_2::MyState
    next_event_info::coupled_event_info
    coupled_next::Bool
    coupled::Bool
    coupled_time::couple_info
    method::Int
end

# Set the thinning bound
function update_thin(thin::AffinePoisson, t)

    a = thin.a
    b = thin.a*t + thin.b
    shift = 0.0           # Assumes the updated thinning is time-coupled

    return AffinePoisson(a, b, 0.0, shift)
end

# BPS kernel Moves the state until stochastic time Tmax
function kernel(state, Tmax)
    t, x, v, thin = state
    τr, τb = -log(rand()) / λᵣ, rand(thin)
    is_bounce = τb < τr
    τ = min(τr, τb)
    t_next = t + τ
    
    num_event = 1
    while(t_next < Tmax)
        num_event += 1
        x += v*(t_next - t)
        t = t_next
        if(is_bounce)
            grad = ∇U(x)
            bounce!(v, grad)
        else
            grad = ∇U(x)
            randn!(v)
        end
        thin = get_thin(v, grad, H, 0.0)
        τr, τb = -log(rand()) / λᵣ, rand(thin)
        is_bounce = τb < τr
        τ = min(τr, τb)
        t_next = t + τ
    end
    x += v*(Tmax - t)
    thin = update_thin(thin, Tmax - t)
    t = Tmax
    new_state = (t, x, v, thin)
    return new_state, num_event
end

# Refresh the next event info if no event happens (v not updated)
function get_next_event_info(t₁, t₂, thin_1, thin_2)
    τr₁, τr₂, ref_coupled = coupling_refresh(λᵣ, Δt + t₂ - t₁, couple_mode)
    τb₁, τb₂, b_coupled = coupling_bounce(thin_1, thin_2)
    is_bounce₁ = τb₁ < τr₁
    is_bounce₂ = τb₂ < τr₂
    τ₁ = min(τr₁, τb₁)
    τ₂ = min(τr₂, τb₂)
    return coupled_event_info(τ₁, τ₂, is_bounce₁, is_bounce₂, ref_coupled, false, b_coupled)
end

function coupled_init(x₁::Vector, x₂::Vector, method::Int)
    coupled_event = sampler.init(x₁, x₂)
    state_1, state_2, _, _, _ = coupled_event
    state_1, n_event_1 = kernel(state_1, Δt)

    t₁, x₁, v₁, thin_1 = state_1
    t₂, x₂, v₂, thin_2 = state_2

    new_h₁, new_h₂ = h(x₁), h(x₂)
    new_state_1 = MyState(state_1, new_h₁)
    new_state_2 = MyState(state_2, new_h₂)

    coupled_time = couple_info(t₂, n_event_1, 0)

    # Get next event info
    next_event_info = get_next_event_info(t₁, t₂, thin_1, thin_2)

    new_coupled_state = MyCoupledState(new_state_1, new_state_2, next_event_info, false, false, coupled_time, method)
    return new_coupled_state
end


function coupled_kernel(coupled_state::MyCoupledState)

    state_1, state_2 = coupled_state.state_1, coupled_state.state_2
    next_event_info, coupled_time = coupled_state.next_event_info, coupled_state.coupled_time
    t₁, x₁, v₁, thin_1 = state_1.z
    t₂, x₂, v₂, thin_2 = state_2.z

    next_t₁, next_t₂ = t₁ + Δt, t₂ + Δt

    # Check if next event has coupled in time:
    coupled_time_bounce = next_event_info.b_coupled & next_event_info.bounce₁ & next_event_info.bounce₂
    coupled_time_ref = next_event_info.ref_coupled & !next_event_info.bounce₁ & !next_event_info.bounce₂
    coupled_time_next = (coupled_time_bounce || coupled_time_ref)

    if coupled_state.method == 1
        reclock_time = true
    elseif coupled_state.method == 2
        reclock_time = !coupled_time_next #reclock if not coupled
    elseif coupled_state.method > 2
        reclock_time = (((t₁/Δt) % coupled_state.method) == 0)
    end

    # Refresh the event independent of the past
    if(reclock_time)
        next_event_info = get_next_event_info(t₁, t₂, thin_1, thin_2)
        coupled_next = false
        coupled = false
        coupled_time.num_events_1 += 1
        coupled_time.num_events_2 += 1
    else
        coupled_next = coupled_state.coupled_next
        coupled = coupled_state.coupled
    end
    
    #Check if next event is past the next time
    update_1 = t₁ + next_event_info.τ₁ < next_t₁
    update_2 = t₂ + next_event_info.τ₂ < next_t₂

    state_1 = (t₁, x₁, v₁, thin_1)
    state_2 = (t₂, x₂, v₂, thin_2)

    while update_1 && update_2
        coupled_event = sampler.onestep(state_1, state_2, next_event_info, coupled_next, coupled)
        state_1, state_2, next_event_info, coupled_next, coupled = coupled_event
        t₁, x₁, v₁, thin_1 = state_1
        t₂, x₂, v₂, thin_2 = state_2

        update_1 = t₁ + next_event_info.τ₁ < next_t₁
        update_2 = t₂ + next_event_info.τ₂ < next_t₂

        if(!coupled)
            coupled_time.stoch_time = t₂
            coupled_time.num_events_1 += 1
            coupled_time.num_events_2 += 1
        end
    end

    if update_1
        # Update process 1 until stoch time t_next
        (t₁, x₁, v₁, thin_1), num_event_1 = kernel(state_1, next_t₁)
        if(!coupled)
            coupled_time.num_events_1 += num_event_1
        end
        # Move process 2 and update the thinning to the new position
        x₂ += (next_t₂ - t₂)*v₂
        thin_2 = update_thin(thin_2, next_t₂ - t₂)
        t₂ = next_t₂
        # Get next event info
        next_event_info = get_next_event_info(t₁, t₂, thin_1, thin_2)

    elseif update_2
        # Update process 2 to stoch time t_next
        (t₂, x₂, v₂, thin_2), num_event_2 = kernel(state_2, next_t₂)
        if(!coupled)
            coupled_time.num_events_2 += num_event_2
        end
        x₁ += (next_t₁ - t₁)*v₁
        thin_1 = update_thin(thin_1, next_t₁ - t₁)
        t₁ = next_t₁

        next_event_info = get_next_event_info(t₁, t₂, thin_1, thin_2)
    else
        # If both processess can continue moving adjust the event information and continue.
        x₁ += (next_t₁ - t₁)*v₁
        x₂ += (next_t₂ - t₂)*v₂

        thin_1 = update_thin(thin_1, next_t₁ - t₁)
        thin_2 = update_thin(thin_2, next_t₂ - t₂)

        # Update the tau values for the next event 
        τ₁ = next_event_info.τ₁ - (next_t₁ - t₁)
        τ₂ = next_event_info.τ₂ - (next_t₂ - t₂)
        next_event_info = coupled_event_info(τ₁, τ₂, next_event_info.bounce₁,next_event_info.bounce₂, 
                                            next_event_info.ref_coupled, next_event_info.ref_pos_coupled, next_event_info.b_coupled)

        t₁ = next_t₁
        t₂ = next_t₂
    end

    new_h₁, new_h₂ = h(x₁), h(x₂)
    new_state_1 = MyState((t₁, x₁, v₁, thin_1), new_h₁)
    new_state_2 = MyState((t₂, x₂, v₂, thin_2), new_h₂)

    new_coupled_state = MyCoupledState(new_state_1, new_state_2, next_event_info, coupled_next, coupled, coupled_time, coupled_state.method)
    return new_coupled_state
end

