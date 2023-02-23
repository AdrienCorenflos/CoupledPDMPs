include("../poisson/affine.jl")
using LinearAlgebra: norm

struct PDMP
    """ PDMP 
    Give a function to initialise the sampler and set the one-step function
    """
    init::Function
    onestep::Function
end