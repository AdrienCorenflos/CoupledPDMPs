include("pdmp.jl")

struct BPSstate
    """ State of the BPS
    The BPS algo takes a position and returns the next thinned event.
    To improve computation on the bounce and thinning additional info is stored
    """
    x::Vector
    v::Vector
    ∇Ux::Vector
    thin_prop::AffinePoisson
end

struct BPSinfo
    """ Info on the BPS thinning event
    Store any info on the current state of BPS
    """
    is_event::Bool
    is_refresh::Bool
    acceptance_rate::Float64
end

function bounce(v, grad)
    grad = grad / norm(grad, 2)
    v -=  2 * sum(grad .* v) * grad
    return v
end


function BPS(∇U::Function, H::Matrix, λᵣ::Float64)
    
    function kernel(state::BPSstate)
        x, v, thin = state.x, state.v, state.thin_prop

        τ = rand(thin)
        x += τ*v

        grad = ∇U(x)
        is_event = false
        is_refresh = false

        λₑ = max(0., thin.a*τ + thin.b)
        λₜ = λₑ + λᵣ

        if(rand() < λᵣ/λₜ)
            # If refeshment event
            randn!(v)
            a = v'*H*v
            b = v'*grad
            is_event = true
            is_refresh = true
            acc_rate = 1.
        else
            # Thinning
            switchrate = v'*grad
            acc_rate = switchrate/λₑ          

            if(λₑ < switchrate -1e-10)
                println("Error in thinning")
            end

            if(rand()*λₑ <= switchrate)
                v = bounce(v, grad)
                a = v'*H*v
                b = -switchrate
                is_event = true
            else
                a = thin.a
                b = switchrate
            end 
        end

        newstate = BPSstate(x, v, grad, AffinePoisson(a,b,λᵣ))
        info = BPSinfo(is_event, is_refresh, acc_rate)

        return newstate, info
    end
    function init(position::Vector)
        velocity = randn(length(position))
        grad = ∇U(position)

        # Set the thinning bound
        a = velocity'*H*velocity
        b = velocity'*grad

        return BPSstate(position, velocity, grad, AffinePoisson(a,b,λᵣ))
    end
    return PDMP(init, kernel)
end

