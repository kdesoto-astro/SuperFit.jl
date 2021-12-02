include("../src/SuperFit.jl")

import .SuperFit
using Turing, MCMCChains
using StatsPlots
using CSV, DataFrames

function save_traces(ztf_name::String, filter_band::String)
    stored_model_dir = "../stored_models"
    tracepath = joinpath(stored_model_dir, string(ztf_name,"_1",filter_band))
    ENV["GKSwstype"] = "100"
    println("entered function")
    trace = read(tracepath, Chains)
    CSV.write("full_file.csv", DataFrame(trace))
    println("loaded in trace")
    println(DataFrame(summarystats(trace)))
    plot_all_traces(trace, ztf_name, filter_band)
end

function plot_all_traces(trace, ztf_name, filter_band)
    num_params = 9
    for param_idx in 1:num_params
        chn_param = trace[:,param_idx,:]
        plt = plot(Array(chn_param),
            xlabel = "Iterations",
            ylabel = "Value"
        )
        savefig(plt, string("trace_",ztf_name,"_",filter_band,"_",param_idx,".png"))
    end
end

function save_marginals(ztf_name::String, filter_band::String)
    stored_model_dir = "../../stored_models"
    #tracepath = joinpath(stored_model_dir, string(ztf_name,"_1",filter_band))
    tracepath = joinpath(stored_model_dir, ztf_name)
    ENV["GKSwstype"] = "100"
    println("entered function")
    trace = read(tracepath, Chains)
    #CSV.write("full_file.csv", DataFrame(trace))
    println("loaded in trace")
    println(DataFrame(summarystats(trace)))
    plot_all_marginals(trace, ztf_name, filter_band)
end

function plot_all_marginals(trace, ztf_name, filter_band)
    num_params = 9
    for param_idx in 1:num_params
        chn_param = trace[:,param_idx,:]
        plt = histogram(Array(chn_param))
        savefig(plt, string("../../simulated_plots/marginal_",ztf_name,"_",filter_band,"_",param_idx,".png"))
    end
end

function show_summary_stats(filename::String)
    trace = read(filename, Chains)
    sumstats = DataFrame(summarystats(trace))
    return sumstats
end

function flux_map(t, A, beta, gamma_1, gamma_2, gamma_switch, t_0, tau_rise, tau_fall, extra_sigma)
    gamma = gamma_1
    if gamma_switch > 0.6666
        gamma = gamma_2
    end
    phase = t - t_0
    f = A / (1. + exp(-phase / tau_rise)) * (1. - beta * gamma) * exp((gamma - phase) / tau_fall)
    if phase < gamma
        f = A / (1. + exp(-phase / tau_rise)) * (1. - beta * phase)
    end
    return -2.5*log10(f) + 22.
end

function plot_fits(ztf_name::String, filter_band::String)
    ENV["GKSwstype"] = "100"
    all_lightcurve_data = "../ztf_data/filtered_data_20"
    stored_model_dir = "../stored_models"
    
    ztf_filepath = joinpath(all_lightcurve_data, string(ztf_name,".txt"))
    println(ztf_filepath)
    ztf_photometry = CSV.read(ztf_filepath, DataFrame, delim=' ')
    println(ztf_photometry)
    if filter_band=="r"
        entries = filter(row -> row.ZTF_filter == "r", ztf_photometry)
    else
        entries = filter(row -> row.ZTF_filter == "g", ztf_photometry)
    end
    times = entries.ZTF_MJD
    #p1 = plot(r_entries.ZTF_MJD, r_entries.ZTF_PSF, seriestype = :scatter, yflip=true, color=1)
    #p2 = plot(g_entries.ZTF_MJD, g_entries.ZTF_PSF, seriestype = :scatter, yflip=true, color=2)
    tracepath = joinpath(stored_model_dir, string(ztf_name,"_1",filter_band))
    println(tracepath)
    trace = MCMCChains.read(tracepath, Chains)
    num_chains = length(trace[1, :A, :])
    println(num_chains)
    all_fluxes = Array{Vector{Float64}}(undef, num_chains)
    for i in 1:num_chains
        single_chain = median(Array(trace[:,:,i]), dims=1)
        println(single_chain)
        model_flux = [flux_map(t, single_chain...) for t in times]
        all_fluxes[i] = model_flux
    end
    plot(times, all_fluxes, yflip=true)
    plot!(times, entries.ZTF_PSF, seriestype=:scatter)
    savefig(string("fluxes_",ztf_name,"_",filter_band,".png"))
end

function all_diagnostics(ztf_name::String)
    for filter_band in ["r", "g"]
        plot_fits(ztf_name, filter_band)
        save_traces(ztf_name, filter_band)
        save_marginals(ztf_name, filter_band)
    end
end

save_marginals("simulated_10", "r")