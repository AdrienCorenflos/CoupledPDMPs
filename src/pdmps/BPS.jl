include("pdmp.jl")

"""
State of the Bouncy Particle Sampler

Additional information regarding the thinning procedure is stored for efficiency.

# Fields
- `skeleton::Skeleton`: The skeleton of the PDMP
- `thinning::AffinePoisson`: The proposal for thinning 
"""
struct BPSstate
    skeleton::Skeleton
    thinning::AffinePoisson
    refresh::HomogeneousPoisson
end

struct BPSDiscreteState
    current::Skeleton
    event_vec::Vector{BPSstate}
    h::Any
end

function rand(state::BPSstate)
    """ Make a proposal
    Return if refreshment event
    """
    thin = rand(state.thinning)
    ref = rand(state.refresh)
    τ = min(thin, ref)
    refresh = ref < thin
    
    return refresh, τ
end

function move_linear(skel::Skeleton, τ)
    t = skel.t + τ
    x = skel.x + τ*skel.v
    v = skel.v
    return t, x, v
end

function move(current::Skeleton, event_vec, Δt)

    t = current.t
    t_new = current.t + Δt
    
    while( t < t_new )
        time_to_next_event = event_vec[1].skeleton.t - t
        Δr = t_new - t
        if(time_to_next_event < Δr)
            current_event = popfirst!(event_vec)
            current = current_event.skeleton
        else
            current = Skeleton(move_linear(current, Δr)...)
        end
        t = current.t
    end
    
    return current
end

function bounce!(v, grad)
    nrm = norm(grad, 2)
    grad = grad / nrm
    v[:] = v - 2 * sum(grad .* v) * grad
end

function bounce_kernel(rng, state::BPSstate, H, grad, refresh, τ)
    v = copy(state.skeleton.v)
    if(refresh)
        randn!(rng,v)
        a, b = v'*H*v, v'*grad
        event = true
    else
        λ = v'*grad
        λᵤ = rate(state.thinning, τ)
        if(λᵤ < λ -1e-10)
            println("---------------------------")
            println("Error in thinning")
            println(λ/λᵤ)
            println("---------------------------")
        end

        if(rand(rng)*λᵤ <= λ)
            bounce!(v, grad)
            a, b = v'*H*v, -λ
            event = true
        else
            a, b = state.thinning.a, λ
            event = false
        end
    end
    return a, b, v, event
end

function BPS(∇U::Function, H::Matrix, h::Function, Δt::Float64, λᵣ::Float64)

    function kernel_event(state::BPSstate)
        rng = Random.default_rng()

        refresh, τ = rand(state)
        t, x, _ = move_linear(state.skeleton, τ)
        a, b, v, event = bounce_kernel(rng, state, H, ∇U(x), refresh, τ)

        newskeleton = Skeleton(t, x, v)
        newstate = BPSstate(newskeleton, AffinePoisson(a,b), HomogeneousPoisson(λᵣ))

        return newstate
    end

    function kernel(state::BPSDiscreteState)
        current = state.current
        event_vec = state.event_vec

        while(event_vec[end].skeleton.t < current.t + Δt )
            next_event = kernel_event(event_vec[end])
            event_vec = vcat(event_vec, next_event)
        end

        new_current = move(current, event_vec, Δt)
        newstate = BPSDiscreteState(new_current, event_vec, h(new_current))

        return newstate
    end


    function init(position::Vector, velocity::Vector)
        grad = ∇U(position)

        # Set the thinning bound
        a = velocity'*H*velocity
        b = velocity'*grad

        current = Skeleton(0., position, velocity)
        nextevent = kernel_event(BPSstate(current, AffinePoisson(a,b), HomogeneousPoisson(λᵣ)))
        newstate = BPSDiscreteState(current, [nextevent], h(current))
        return newstate
    end
    
    return DPDMP(init, kernel, kernel_event)
end

