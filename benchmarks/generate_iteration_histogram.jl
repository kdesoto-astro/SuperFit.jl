using MCMCChains
using DynamicPPL
using SuperFit
using Plots

ENV["GKSwstype"] = "100"
"""
path = "../../stored_models_hists"
all_files = readdir(path; join=true)
all_iterations = Vector{Int64}()
for file in all_files
    if occursin("sim", file)
        opened_file = read(file, Chains)
        num_iters = size(opened_file)[1]
        println(num_iters)
        push!(all_iterations, num_iters)
    end
end
histogram(all_iterations, bins=20)
savefig("histogram_all.png")
"""
npoints = [10, 20, 30, 40, 50, 60, 70, 80]
lower_quartile = [25.23, 27.62, 32.74, 37.37, 37.39, 43.11, 48.71, 52.55]
median = [41.49, 32.83, 39.92, 45.58, 42.76, 50.05, 69.40, 63.03]

plot(npoints, lower_quartile)
plot!(npoints, median)
savefig("npoints_dependence.png")


    