include("pdmp.jl")

struct BPSinfo
    """ Info on the BPS thinning event
    Store any info on the current state of BPS
    """
    event::Bool
    refresh::Bool
end

struct BPSstate
    """ State of the BPS
    The BPS algo takes a position and returns the next thinned event.
    To improve computation on the bounce and thinning additional info is stored
    """
    skeleton::Skeleton
    thin_prop::AffinePoisson
    refresh::HomogeneousPoisson
end

function rand(state::BPSstate)
    """ Make a thinnign proposal
    Return next time and if refreshment event
    """
    τₑ = rand(state.thin_prop)
    τᵣ = rand(state.refresh)
    return τᵣ < τₑ, min(τₑ, τᵣ)
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
            λᵤ = rate(thin, τ)
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
                a, b = state.thin_prop.a, λ
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

        a, b, event = bps_internal!(refresh, v, ∇U(x), state.thin_prop, τ)

        newskeleton = Skeleton(t, x, v)
        newstate = BPSstate(newskeleton, AffinePoisson(a,b), HomogeneousPoisson(λᵣ))

        info = BPSinfo(event, refresh)

        return newstate, info
    end

    function init(position::Vector)
        velocity = randn(length(position))
        grad = ∇U(position)

        # Set the thinning bound
        a = velocity'*H*velocity
        b = velocity'*grad

        newskeleton = Skeleton(0., position, velocity)
        return BPSstate(newskeleton, AffinePoisson(a,b), HomogeneousPoisson(λᵣ))
    end
    
    return PDMP(init, kernel)
end

