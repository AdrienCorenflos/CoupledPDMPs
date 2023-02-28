using Distributions
using Random
import Distributions:
    @check_args, logpdf, rand, rand!, insupport, cdf, quantile, minimum, maximum, partype

abstract type PoissonProcess <: ContinuousUnivariateDistribution end


"""
    rate(d::PoissonProcess, t)
Value of the rate function at time t.
"""
rate(d::PoissonProcess, t::Real) = error("rate function not implemented for $(typeof(d))")