function bounce(x::Vector, v::Vector)
    e = sign.(x)
    e = e / norm(e,2)
    v_out = v - 2 * sum(e .* v) * e
    return v_out
end

function bounce_intensity(x::Vector, v::Vector)
    return max(0, sum(sign.(x).*v))
end

function thin_bounce(x::Vector, v::Vector)#, H::Matrix)
    upper = sum(abs.(v))
    tau = 0
    while true
        tau = tau + expon_rvs(upper)
        u = rand()
        a = bounce_intensity(x + tau * v, v)
        if u < a / upper
            return tau
        end
    end
end

function next_bounce_event(x_1::Vector, v_1::Vector, x_2::Vector, v_2::Vector)
    bounce_1 = thin_bounce(x_1, v_1)

    # If coupled return the same bounce
    if (maximum(abs.(x_1 - x_2)) < 1e-10) && (maximum(abs.(v_1 - v_2)) < 1e-10)
        return bounce_1, bounce_1
    else
        bounce_2 = thin_bounce(x_2, v_2)
        return bounce_1, bounce_2
    end
end

function next_refresh_event(t_1, t_2, ref, dt)
    dt = t_2 + dt - t_1
    return coupling_exp(ref, ref, dt)
end

function refresh_velocity_couple_position(tau, x_1::Vector, x_2::Vector)
    v_1, v_2 = reflection_max_couple(x_1, x_2, tau)
    v_1 = (v_1 - x_1)/tau
    v_2 = (v_2 - x_2)/tau
    return v_1, v_2
end

function bps_coupled(x_1::Vector, v_1::Vector, x_2::Vector, v_2::Vector, dt, ref=1, max_iter = 100)
    d = length(x_1)
    t_2 = 0
    t_1 = 0
    t_2_s = fill(0., max_iter)
    t_1_s = fill(0., max_iter)
    xs_1 = fill(0., max_iter, d)
    xs_2 = fill(0., max_iter, d)
    xs_1[1,:] = x_1; xs_2[1,:] = x_2
  
    is_time_coupled = is_position_coupled = is_velocity_coupled = false
  
    it = 1
    while (!is_time_coupled || !is_position_coupled || !is_velocity_coupled) || (it < max_iter) 
        it = it + 1
        if(it > max_iter)
            break
        end

        Random.seed!(it)
        ## Refresh to match time
        next_refresh = next_refresh_event(t_1, t_2, ref, dt)
        
        ## Sample normal bounce
        next_bounce = next_bounce_event(x_1, v_1, x_2, v_2)
        
        tau = min.(next_bounce, next_refresh)
        
        ## Update positions and times
        t_1 += tau[1]
        t_2 += tau[2]
        
        x_1 += tau[1] * v_1
        x_2 += tau[2] * v_2
        
        is_velocity_coupled = false
        is_time_coupled = (abs(t_1 - dt - t_2) < 1e-10)
        
        ## Update Velocity
        
        # If refreshment
        if(tau[1] == next_refresh[1] && tau[2] == next_refresh[2])
            # If time is coupled try couple the velocities to match positions
            if(is_time_coupled)
                Random.seed!(it+1)
                ## Prefetch next refreshment velocity
                next_refresh = next_refresh_event(t_1, t_2, ref, dt)
                
                ## Couple the next position
                v_1, v_2 =  refresh_velocity_couple_position(next_refresh[1], x_1, x_2)
            else
                v_1, v_2 = reflection_max_couple(fill(0,d), fill(0,d), 1)
            end
        
            # position coupled, try couple velocities (not in algo)
            if is_position_coupled
                v_1, v_2 =  reflection_max_couple(zeros(d), zeros(d), 1)
                is_velocity_coupled = true
            end
        end
        # refresh and bounce
        if tau[1] == next_bounce[1] && tau[2] == next_refresh[2]
            v_1 = bounce(x_1, v_1)
            v_2 = randn(d)
        end
        
        # bounce and refresh
        if tau[1] == next_refresh[1] && tau[2] == next_bounce[2]
            v_2 = bounce(x_2, v_2)
            v_1 = randn(d)
        end
        
        # bounce and bounce
        if tau[1] == next_bounce[1] && tau[2] == next_bounce[2]
            v_1 = bounce(x_1, v_1)
            v_2 = bounce(x_2, v_2)
        end
        
        is_position_coupled = max(norm(x_1 - x_2)) < 1e-10
        
        t_1_s[it] = t_1
        xs_1[it,:] = x_1
        t_2_s[it] = t_2
        xs_2[it,:] = x_2
    end
    return (t_1_s, xs_1, t_2_s, xs_2, dt)
end