include("../poisson/affine.jl")
include("../poisson/homogeneous.jl")
using LinearAlgebra: norm

abstract type PDMPState end

struct Skeleton
    t::Float64
    x::Vector
    v::Vector
end 

struct PDMP
    """ PDMP 
    Give a function to initialise the sampler and set the one-step function
    """
    init::Function
    onestep::Function
end

struct DPDMP
    """ Discrete kernel PDMP 
    
    """
    init::Function
    onestep::Function
    onestep_event::Function
end

