function expon_rvs(rate)
    return -log(rand())/rate
end

# Couple Exp variables
function coupling_exp(rate1, rate2, dt)
    rp1() = -log(rand())/rate1
    rp2() = -log(rand())/rate2 + dt

    logdp1(x) = logpdf(Exponential(rate1), x)
    logdp2(x) = logpdf(Exponential(rate2), x - dt)

    X = rp1()
    Y = copy(X)
    acc = (rand() < exp(logdp2(X) - logdp1(X)))

    while !acc
        Y = rp2()
        V = rand()
        acc = (exp(logdp1(Y) - logdp2(Y)) < V)
    end
    Y = Y - dt
    return X, Y
end

function expon_lin(a, b)
    u = rand()
    if a < 0. && b > 0.
        return -a/b + sqrt(-2*log(u)/b)
    elseif a > 0. && b < 0.
        if -a^2/b + a^2/(2*b) >= -log(u)
            return -a/b - sqrt(a/b^2 - 2*log(u)/b)
        else
            return Inf
        end
    elseif a >= 0. && b > 0.
        return -a/b + sqrt(a/b^2 - 2*log(u)/b)
    elseif a > 0. && b == 0
        return -log(u)/a
    else
        return Inf
    end
end