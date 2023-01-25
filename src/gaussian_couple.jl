function reflection_max_couple(m_1::Vector, m_2::Vector, sigma)
    d = length(m_1)
    z = (m_1 - m_2) / sigma
    e = z / norm(z)

    V = randn(d)
    log_U = log(rand())

    multinormal = MvNormal(fill(0, d), fill(1,d))

    if sum(logpdf(multinormal, V)) + log_U < sum(logpdf(multinormal, V + z))
        W = V + z
    else
        W = V - 2 * sum(V .* e) * e
    end
    X = m_1 + sigma*V
    Y = m_2 + sigma*W

    coupled = sum(norm(X - Y)) < 1e-14
    return X, Y
end