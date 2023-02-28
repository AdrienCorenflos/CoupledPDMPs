include("../poisson/affine.jl")
include("../poisson/homogeneous.jl")
using LinearAlgebra: norm

struct PDMP
    """ PDMP 
    Give a function to initialise the sampler and set the one-step function
    """
    init::Function
    onestep::Function
end

struct COUPLEDPDMP
    """ COUPLEDPDMP 
    More general allows for coupling.
    """
    init::Function
    onestep::Function
    onestep_couple::Function
end

struct Skeleton
    t::Float64
    x::Vector
    v::Vector
end 

