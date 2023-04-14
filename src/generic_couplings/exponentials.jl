using Random


"""
    coupling(rate, shift, mode="independent")

Generate two samples, `t1` and `t2`, drawn from a maximal coupling between two exponential distributions
with the same rate `rate`, targetting the event `t1 = t2 + shift`. The residual samples (when the coupling fails) are jointly distributed according to `mode`.

## Arguments
- `rate`: The rate parameter of the exponential distribution.
- `shift`: The shift amount that couples `t1` and `t2` together. Must be positive.
- `mode`: One of `"independent"`, `"crn"`, or `"antithetic"`. Determines the type of residual coupling
  used between `t1` and `t2`. `"independent"` produces two independent samples; `"crn"` and `"antithetic"`
  produce correlated and negatively correlated samples, respectively. Default is `"independent"`.

## Returns
A tuple `(t1, t2, coupled)` of two samples, `t1` and `t2`, drawn from two exponential distributions,
and a boolean value `coupled` indicating whether the samples are coupled together, i.e., whether we have `t1 = t2 + shift`. 

## Examples
```julia
julia> coupling(1.0, 0.5)
(0.83, 0.33, true)

julia> coupling(1.0, 0.5, "crn")
(0.5991642448732429, 0.7317536655848758, false)

julia> coupling(1.0, 0.5, "antithetic")
(0.6934063052821695, 0.3065936947178305, false)

julia> coupling(1.0, 0.5, "independent")
(0.4046461973224223, 0.13215386221532302, false)
"""
function coupling(rate::U, shift::U, mode::T = "independent")::Tuple{U, U, Bool} where {U<:AbstractFloat, T<:AbstractString}
    mixture_weight = exp(-rate * shift)
    u = rand()

    coupled = u < mixture_weight
    if u < mixture_weight
        log_v = log(rand())
        t_1 = shift - log_v / rate
        t_2 = t_1
    else
        if mode == "independent"
            v = rand()
            w = rand()
        elseif mode == "crn"
            v = rand()
            w = 0. + v  # to avoid reference and force copy
        elseif mode == "antithetic"
            v = rand()
            w = 1. - v
        else
            error("Invalid mode.")
        end
        
        v *= (1 - mixture_weight)
        v = 1 - v

        # This should perhaps be done in logspace
        w *= 1 - mixture_weight
        w /= (exp(rate * shift) - 1)
        w = exp(-rate * shift) - w

        log_v = log(v)
        log_w = log(w)
        t_1 = -log_v / rate
        t_2 = -log_w / rate
    end

    return t_1, t_2 - shift, coupled
end