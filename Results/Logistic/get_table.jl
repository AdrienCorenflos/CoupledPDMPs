using JLD2

function assign_to_global(dict::Dict{String, Array{Float64}})
    for (key, value) in dict
        @eval global $(Symbol(key)) = $(value)
    end
end

@load "Results/Logistic/raw_results/BPS_K37_l4.0_t8.0.jld2"

assign_to_global(matrices_dict)

@load "Results/Logistic/raw_results/BPS_ref.jld2"

assign_to_global(matrices_dict_ref)


println("\n COMP (total grads used)")
println(mean(comp[:,4,1] + comp[:,2,1]+ comp[:,3,1]))
println(mean(comp[:,4,2] + comp[:,2,2]+ comp[:,3,2]))
println(mean(comp[:,4,3] + comp[:,2,3]+ comp[:,3,3]))

println("\n VARIANCE")
println(sum(var(h_1[:,1],dims =1)) + sum(var(h_2[:,:,1],dims =1)))
println(sum(var(h_1[:,2],dims =1)) + sum(var(h_2[:,:,2],dims =1)))
println(sum(var(h_1[:,3],dims =1)) + sum(var(h_2[:,:,3],dims =1)))

println("\n Ineff h = h1 + h2")
Ineff1 = mean(comp[:,4,1] + comp[:,2,1]+ comp[:,3,1])*(sum(var(h_1[:,1],dims =1)) + sum(var(h_2[:,:,1],dims =1))) / ( mean(refcomp[:,1]) * (sum(var(refh_1[:,1],dims =1))+sum(var(refh_2[:,:,1],dims =1))))
Ineff2 = mean(comp[:,4,2] + comp[:,2,2]+ comp[:,3,2])*(sum(var(h_1[:,2],dims =1)) + sum(var(h_2[:,:,2],dims =1))) / ( mean(refcomp[:,2]) * (sum(var(refh_1[:,2],dims =1))+sum(var(refh_2[:,:,2],dims =1))))
Ineff3 = mean(comp[:,4,3] + comp[:,2,3]+ comp[:,3,3])*(sum(var(h_1[:,3],dims =1)) + sum(var(h_2[:,:,3],dims =1))) / ( mean(refcomp[:,3]) * (sum(var(refh_1[:,3],dims =1))+sum(var(refh_2[:,:,3],dims =1))))
println(Ineff1)
println(Ineff2)
println(Ineff3)


@load "Results/Logistic/raw_results/BOOM_K10_l3.0_t8.0.jld2"

assign_to_global(matrices_dict)

@load "Results/Logistic/raw_results/BOOM_ref.jld2"

assign_to_global(matrices_dict_ref)

println("\n COMP")
println(mean(comp[:,4,1] + comp[:,2,1]+ comp[:,3,1]))
println(mean(comp[:,4,2] + comp[:,2,2]+ comp[:,3,2]))
println(mean(comp[:,4,3] + comp[:,2,3]+ comp[:,3,3]))

println("\n VARIANCE")
println(sum(var(h_1[:,1],dims =1)) + sum(var(h_2[:,:,1],dims =1)))
println(sum(var(h_1[:,2],dims =1)) + sum(var(h_2[:,:,2],dims =1)))
println(sum(var(h_1[:,3],dims =1)) + sum(var(h_2[:,:,3],dims =1)))

println("\n Ineff h = h1 + h2")
Ineff1 = mean(comp[:,4,1] + comp[:,2,1]+ comp[:,3,1])*(sum(var(h_1[:,1],dims =1)) + sum(var(h_2[:,:,1],dims =1))) / ( mean(refcomp[:,1]) * (sum(var(refh_1[:,1],dims =1))+sum(var(refh_2[:,:,1],dims =1))))
Ineff2 = mean(comp[:,4,2] + comp[:,2,2]+ comp[:,3,2])*(sum(var(h_1[:,2],dims =1)) + sum(var(h_2[:,:,2],dims =1))) / ( mean(refcomp[:,2]) * (sum(var(refh_1[:,2],dims =1))+sum(var(refh_2[:,:,2],dims =1))))
Ineff3 = mean(comp[:,4,3] + comp[:,2,3]+ comp[:,3,3])*(sum(var(h_1[:,3],dims =1)) + sum(var(h_2[:,:,3],dims =1))) / ( mean(refcomp[:,3]) * (sum(var(refh_1[:,3],dims =1))+sum(var(refh_2[:,:,3],dims =1))))
println(Ineff1)
println(Ineff2)
println(Ineff3)