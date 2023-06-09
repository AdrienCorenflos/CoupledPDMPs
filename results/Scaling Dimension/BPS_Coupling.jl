include("../../src/pdmps/PDMP_discrete.jl")

using Distributions
using Random
using LinearAlgebra
using StatsBase

## Information on the problem sampled.
function h(x)
    return x[1:2]
end

function simulation_study(d_values, Δt_values, M_values, num_runs)
    # Initialize result containers
    estimators = Array{Float64}(undef, num_runs, length(d_values), length(Δt_values), length(M_values), 2)
    coupling_times = Array{Float64}(undef, num_runs, length(d_values), length(Δt_values), length(M_values), 2)
    coupling_events = Array{Int64}(undef, num_runs, length(d_values), length(Δt_values), length(M_values), 2)
    computation_times = Array{Int64}(undef, num_runs, length(d_values), length(Δt_values), length(M_values), 2)


    # Loop over the initial distribution type (offset or in stationarity)
    for (sim_ind, offset) in enumerate([true, false])
        # Outer loop over d values
        for (d_index, d) in enumerate(d_values)
            λᵣ = 1.0
            # Middle loop over Δt values
            for (Δt_index, Δt) in enumerate(Δt_values)
                # Inner loop over M values
                for (M_index, M) in enumerate(M_values)
                    # Run the simulation for each combination of λᵣ, Δt, and M
                    println("Sim: λᵣ ", λᵣ," Δt ", Δt," M ", M, " d", d)

                    # Define the problem
                    H = Matrix{Float64}(I,d,d)
                    function ∇U(x::Vector)
                        H*x 
                    end

                    # Define the sampler
                    sampler = BPS_coupling(∇U, H, Δt, λᵣ, "antithetic")
                    coupled_bps = DiscretePDMPCoupling(sampler,  h)

                    for run in 1:num_runs
                        Random.seed!(run)
                        println(run)
                        if(offset)
                            Σ_I = Symmetric(rand(Wishart(d, Matrix{Float64}(I,d,d))))
                            Σ_I_chol = cholesky(Σ_I).L
                            x1 = Σ_I_chol*(randn(dim(H)).+1.0); x2 = Σ_I_chol*(randn(dim(H)).+1.0)
                        else
                            x1 = randn(dim(H)); x2 = randn(dim(H))
                        end

                        coupled_state = coupled_bps.init(x1, x2, 1, M)
                        i = 1
                        while true
                            if(coupled_state.coupled)
                                break
                            end
                            coupled_state = coupled_bps.kernel(coupled_state)
                            i+=1
                        end
                        cs = coupled_state.coupled_status
                        coupling_times[run, d_index, Δt_index, M_index, sim_ind] = cs.stoch_time
                        coupling_events[run, d_index, Δt_index, M_index, sim_ind] = cs.num_events_2
                        computation_times[run, d_index, Δt_index, M_index, sim_ind] = cs.num_events_1+cs.num_events_2+cs.num_event_coupled
                    end
                end
            end
        end
    end
    return coupling_times, coupling_events, computation_times
end

# Define the values of d, Δt, and M to iterate over 
#d_values = [13, 25, 37, 50] # Original values used in Pierre's unbiased MCMC
d_values = [32, 64, 128, 256]
Δt_values = [1., 2., 3.]
M_values = [5]

# Set the number of simulation runs
#num_runs = 500
#coupling_times, coupling_events, computation_times = simulation_study(d_values, Δt_values, M_values, num_runs)

# Save results for later
using JLD2
# Create a dictionary to store the matrices
#matrices_dict = Dict("computation_times" => computation_times,
# "coupling_times" => coupling_times, "coupling_events" => coupling_events)

 # Save the matrices to the file
#@save "results/scaling_with_offset.jld2" matrices_dict

@load "results/scaling_with_offset.jld2" 
coupling_times =  matrices_dict["coupling_times"]
coupling_events  =  matrices_dict["coupling_events"]
computation_times =  matrices_dict["computation_times"]


using Plots
using StatsPlots: boxplot

for (Δt_index, Δt) in enumerate(Δt_values)
    # Extract the corresponding data for the current dt value
    data = coupling_times[:, :, Δt_index, 1, 1]
    
    if Δt_index == 1
        p = plot(d_values, mean(data,dims = 1)', xlabel="d", ylabel="Avg Couple Time",
                label ="Δt = $Δt", legend = false, title = "Starting in target")
    else
        plot!(d_values, mean(data,dims = 1)', xlabel="d", ylabel="Avg Couple Time",
        label="Δt = $Δt", legend = true)
    end
end
plot(p)

for (Δt_index, Δt) in enumerate(Δt_values)
    # Extract the corresponding data for the current dt value
    data = coupling_times[:, :, Δt_index, 1, 2]
    
    if Δt_index == 1
        p = plot(d_values, mean(data,dims = 1)', xlabel="d", ylabel="Avg Couple Time",
                label ="Δt = $Δt", legend = false, title = "Starting in offset")
    else
        plot!(d_values, mean(data,dims = 1)', xlabel="d", ylabel="Avg Couple Time",
        label="Δt = $Δt", legend = true)
    end
end
plot(p)

