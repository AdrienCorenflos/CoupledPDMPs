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
end

function rand(state::BPSstate)
    """ Make a proposal
    Return if refreshment event
    """
    τ = rand(state.thinning)
    λₜ = rate(state.thinning, τ)
    λᵣ = state.thinning.c
    refresh = rand()*λₜ < λᵣ

    return refresh, τ
end

function bounce!(v, grad)
    nrm = norm(grad, 2)
    grad = grad / nrm
    v[:] = v - 2 * sum(grad .* v) * grad
end

function BPS(∇U::Function, H::Matrix, λᵣ::Float64)
    
    function bps_internal!(refresh, v, grad, thin, τ)
        if(refresh)
            randn!(v)
            a, b = v'*H*v, v'*grad
            event = true
        else
            λ = v'*grad
            λᵤ = rate(thin, τ) - λᵣ
            if(λᵤ < λ -1e-10)
                println("---------------------------")
                println("Error in thinning")
                println(λ/λᵤ)
                println("---------------------------")
            end

            if(rand()*λᵤ <= λ)
                bounce!(v, grad)
                a, b = v'*H*v, -λ
                event = true
            else
                a, b = thin.a, λ
                event = false
            end
        end
        return a, b, event
    end

    function kernel(state::BPSstate)
        t, x, v = state.skeleton.t, state.skeleton.x, state.skeleton.v

        refresh, τ = rand(state)
        x += τ*v
        t += τ

        a, b, event = bps_internal!(refresh, v, ∇U(x), state.thinning, τ)

        newskeleton = Skeleton(t, x, v)
        newstate = BPSstate(newskeleton, AffinePoisson(a,b,λᵣ))

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
    
    return PDMP(init, kernel)
end

