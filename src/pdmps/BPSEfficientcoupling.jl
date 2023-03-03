include("BPS_Efficient.jl")

struct BPScoupledstate
    """ State of the BPS
    The BPS algo takes a position and returns the next thinned event.
    To improve computation on the bounce and thinning additional info is stored
    """
    state1::BPSstate
    state2::BPSstate
    Δt::Float64
    coupled::Bool
end

function rand(state::BPScoupledstate)
    """ Make a coupled proposal
    Return next time and if refreshment event
    """
    state1, state2 = state.state1, state.state2
    λᵣ = state1.thinning.c                         # Assumes the same ref rate for both

    τ1, τ2, coupled_time = thorisson(state1.thinning, state2.thinning)
    λ1, λ2 = rate(state1.thinning, τ1), rate(state2.thinning, τ2)
    τ2 = τ2 - state2.thinning.shift

    refresh1, refresh2, coupled = thorisson(Bernoulli(λᵣ/λ1), Bernoulli(λᵣ/λ2)) # Lazy coupling

    return coupled_time, refresh1, refresh2, τ1, τ2
end

function move_ΔtN(skel::Skeleton, Δt, N)
    skel_vec = Vector{Skeleton}(undef, N)
    for i in 1:N
        skel = Skeleton(skel.t+Δt, skel.x+Δt*skel.v, skel.v)
        skel_vec[i] = skel
    end
    return skel_vec, skel.t
end

function update_states(state::BPScoupledstate, τ1, τ2)
    """ Evolve the state by tau assuming no event
    In future return h evaluated at Delta along the trajectory
    """
    Δt = state.Δt
    skel1, skel2 = state.state1.skeleton, state.state2.skeleton
        
    x1 = skel1.x + τ1*skel1.v; x2 = skel2.x + τ2*skel2.v
    v1 = skel1.v; v2 = skel2.v
    t1 = skel1.t + τ1; t2 = skel2.t + τ2

    return t1, x1, v1, t2, x2, v2
end

function BPS_coupling(∇U::Function, H::Matrix, λᵣ::Float64)

    function bps_internal!(rng, refresh, v, grad, thin, τ)
        if(refresh)
            randn!(rng,v)
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

    function kernel_coupling(coupledstate::BPScoupledstate)
        Δt = coupledstate.Δt

        coupled_time, refresh1, refresh2, τ1, τ2 = rand(coupledstate)
                
        t1, x1, v1, t2, x2, v2 = update_states(coupledstate, τ1, τ2)
        
        grad1 = ∇U(x1); grad2 = ∇U(x2)

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
            b1 = v1'*grad1; b2 = v2'*grad2 
            event1 = true; event2 = true
        else
            a1, b1, event1 = bps_internal!(rng,refresh1,v1,grad1,coupledstate.state1.thinning,τ1)
            a2, b2, event2 = bps_internal!(current_rng,refresh2,v2,grad2,coupledstate.state2.thinning,τ2)
        end

        newskeleton1 = Skeleton(t1, x1, v1)
        newstate1 = BPSstate(newskeleton1, AffinePoisson(a1,b1,λᵣ))

        newskeleton2 = Skeleton(t2, x2, v2)
        newstate2 = BPSstate(newskeleton2, AffinePoisson(a2,b2,λᵣ, t2+Δt-t1))

        coupled = (sum(abs.(x1.-x2)) < 1e-10) & (sum(abs.(v1.-v2)) < 1e-10) & (t1 - Δt - t2  < 1e-10)
        
        newcoupledstate = BPScoupledstate(newstate1, newstate2, Δt, coupled)

        return newcoupledstate
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

    function kernel(state::BPSstate)
        t, x, v = state.skeleton.t, state.skeleton.x, state.skeleton.v

        refresh, τ = rand(state)
        x += τ*v
        t += τ

        rng = Random.default_rng()
        a, b, event = bps_internal!(rng,refresh, v, ∇U(x), state.thinning, τ)

        newskeleton = Skeleton(t, x, v)
        newstate = BPSstate(newskeleton, AffinePoisson(a,b,λᵣ))

        return newstate
    end

    function init_coupling(position1::Vector, position2::Vector)
        state1 = kernel(init(position1))
        state2 = init(position2)
        Δt = state1.skeleton.t
        return BPScoupledstate(state1, state2, Δt, false)
    end

    return COUPLEDPDMP(init_coupling, kernel, kernel_coupling)
end

