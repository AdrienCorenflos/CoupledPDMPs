using JLD2
using CSV
using DataFrames
using Plots

function assign_to_global(dict::Dict{String, Array{Float64}})
    for (key, value) in dict
        @eval global $(Symbol(key)) = $(value)
    end
end

@load "Results/Logistic/raw_results/BPS_tuning.jld2"
assign_to_global(matrices_dict_tuning)

res_ineff = nGradTotal .* (var_h1 + var_h2)

p_BPS = Plots.plot(l_values, res_ineff[1,:] |> Vector,
    xlabel="Refreshment λ", ylabel="ineff", yscale = :log10,
    label ="Δ = $(dt_values[1])", title = "BPS")

for i in 2:length(dt_values)
    Plots.plot!(l_values, res_ineff[i,:] |> Vector,
    xlabel="Refreshment λ", ylabel="ineff", yscale = :log10, legend=false,
    label ="Δ = $(dt_values[i])")
end
display(p_BPS)


@load "Results/Logistic/BOOM_tuning.jld2"
assign_to_global(matrices_dict_tuning)

res_ineff = nGradTotal .* (var_h1 + var_h2)

p_BOOM = Plots.plot(l_values, res_ineff[1,:] |> Vector,
    xlabel="Refreshment λ", ylabel="ineff", yscale = :log10,
    label ="Δ = $(dt_values[1])", title = "Boomerang")

for i in 2:length(dt_values)
    Plots.plot!(l_values, res_ineff[i,:] |> Vector,
    xlabel="Refreshment λ", ylabel="ineff", yscale = :log10, legend=false,
    label ="Δ = $(dt_values[i])")
end
display(p_BOOM)




