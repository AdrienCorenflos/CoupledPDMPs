
@doc raw"""
    rhee_glynn(coupled_kernel, coupled_state, k, m[, return_classical_estimator])

Computes the Rhee & Glynn estimator for the coupled kernel `coupled_kernel` after `k` burnin updates, using `m` samples.
If `return_classical_estimator` is `true`, also returns the classical estimator to compare with the bias corrected version.

Formally, for the two τ-coupled shifted chains ``Xₖ₊₁`` and ``Yₖ``, and a test function ``h``, the Rhee & Glynn estimator is defined as

```math
\hat{h}_{k:m} = \frac{1}{m - k + 1} \sum_{i=k}^m h(X_i) + \sum_{i=k+1}^{\tau - 1} \{h(X_i) - h(Y_{i-1}\}.

```
where the classical estimator is given by the first term.

In this implementation, we choose to consider ``h(X_i)`` as part of the Markov chain: the coupled_state will own a value of ``h`` evaluated on its chain. 
This is because we mostly work with PDMPs, so that integrating the test function over the trajectory is better left to the sampler itself which knows its dynamics.

For example, if our kernel is a PDMP, and we use time-integrated estimators, the state should look like

```julia
struct MyState
    skeleton::Skeleton  # a sampler-specific structure that will be useful in returning the next state
    h::Any              # the test function h integrated between ``t - Δ`` and ``t``
    t::Float64          # the current time
end
```

The `CoupledState`, on the other hand, will be a container for the two states, and will be passed to the kernel.

```julia
struct MyCoupledState
    state1::MyState
    state2::MyState
    coupled::Bool     # whether the two states are coupled or not
end
```

# Arguments

- `coupled_kernel::Function`: the kernel that will be used to generate the Markov chain. It should take a `CoupledState` as input and return a new `CoupledState`.
- `coupled_state::CoupledState`: the initial state of the Markov chain. 
    It should contain two states preemptively shifted by 1. In the context of PDMPs, `state1` should be at time `Δ` while `state2` should be at time `0`.
    These cannot be coupled just yet, so `coupled_state.coupled` should be `false`.
- `k::Int`: the number of burnin updates.
- `m::Int`: the number of samples.
- `return_classical_estimator::Bool = false`: whether to return the classical estimator as well.

# Returns

- `estimator::Float64`: the Rhee & Glynn estimator.
- `classical_estimator::Float64`: the classical estimator. Only returned if `return_classical_estimator` is `true`.

"""
function rhee_glynn(
    coupled_kernel::Function,
    coupled_state,
    k::Int,
    m::Int,
    return_classical_estimator::Bool = false,
)
    if ~(0 <= k <= m)
        throw(ArgumentError("k must be in [1, m]"))
    end

    den = m - k
    coupling_time = Inf
    # burnin
    for i in 1:k
        coupled_state = coupled_kernel(coupled_state)
        if coupled_state.coupled
            coupling_time = min(i, coupling_time)
        end
    end

    i_km = map(x -> x / den, coupled_state.state_1.h)   # classical estimator
    b_k = map(x -> 0.0 * x, i_km)   # bias correction

    i = k
    coupled = coupled_state.coupled
    while i < m || ~coupled
        coupled_state = coupled_kernel(coupled_state)
        h_1, h_2 = coupled_state.state_1.h, coupled_state.state_2.h

        coupled = coupled_state.coupled
        if coupled
            coupling_time = min(i, coupling_time)
        end
        if ~coupled
            factor = min(1, (i + 1 - k) / den)
            b_k = map((u, v, w) -> (u + factor * (v - w)), b_k, h_1, h_2)
        end
        if i < m
            i_km = map((u, v) -> (u + v / den), i_km, h_1)
        end
        i += 1
    end

    h_km = map((u, v) -> u + v, i_km, b_k)

    if return_classical_estimator
        return coupling_time, h_km, i_km
    else
        return coupling_time, h_km
    end

end


