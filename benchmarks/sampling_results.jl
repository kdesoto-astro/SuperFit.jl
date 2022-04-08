import Pkg
Pkg.activate("..")
Pkg.instantiate()
using Turing, MCMCChains
using StatsPlots, Plots
using StatsBase
using LsqFit
using CSV, DataFrames
using Distributions
using SuperFit

function save_traces(ztf_name::String, filter_band::String)
    #stored_model_dir = "../stored_models"
    #tracepath = joinpath(stored_model_dir, string(ztf_name,"_1",filter_band))
    tracepath = ztf_name
    ENV["GKSwstype"] = "100"
    println("entered function")
    trace = read(tracepath, Chains)
    #CSV.write("full_file.csv", DataFrame(trace))
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
        #savefig(plt, string("trace_",ztf_name,"_",filter_band,"_",param_idx,".png"))
        savefig(plt, string("wonky_trace", param_idx, ".png"))
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

function plot_fits(ztf_name::String)
    ENV["GKSwstype"] = "100"
    all_lightcurve_data = "../../project-kdesoto-psu/ztf_data/filtered_data_20"
    stored_model_dir = "../../ZTF_fits"
    
    ztf_filepath = joinpath(all_lightcurve_data, string(ztf_name,".txt"))
    println(ztf_filepath)
    ztf_photometry = CSV.read(ztf_filepath, DataFrame, delim=' ')
    println(ztf_photometry)
    plot()
    for filter_band in ["r", "g"]
        if filter_band=="r"
            entries = filter(row -> row.ZTF_filter == "r", ztf_photometry)
        else
            entries = filter(row -> row.ZTF_filter == "g", ztf_photometry)
        end
        times = entries.ZTF_MJD
        #p1 = plot(r_entries.ZTF_MJD, r_entries.ZTF_PSF, seriestype = :scatter, yflip=true, color=1)
        #p2 = plot(g_entries.ZTF_MJD, g_entries.ZTF_PSF, seriestype = :scatter, yflip=true, color=2)
        tracepath = joinpath(stored_model_dir, string(ztf_name,"_",filter_band,".jls"))
        println(tracepath)
        trace = MCMCChains.read(tracepath, Chains)
        num_chains = length(trace[1, :A, :])
        println(num_chains)
        all_fluxes = Array{Vector{Float64}}(undef, num_chains)
        for i in 1:num_chains
            A, beta, gamma, t_0, tau_rise, tau_fall, extra_sigma = median(Array(trace[:,:,i]), dims=1)
            single_chain = (;A, beta, gamma, t_0, tau_rise, tau_fall, extra_sigma)
            println(single_chain)
            model_flux = Vector{Float64}()
            for t in times
                push!(model_flux,  -2.5*log10(SuperFit.flux_map(t, single_chain)) + 22.)
            end
            all_fluxes[i] = model_flux
        end
        plot!(times, all_fluxes, yflip=true)
        plot!(times, entries.ZTF_PSF, seriestype=:scatter)
    end
    savefig(string("fluxes_",ztf_name,"_both" ,".png"))
end

function all_diagnostics(ztf_name::String)
    for filter_band in ["r", "g"]
        plot_fits(ztf_name, filter_band)
        save_traces(ztf_name, filter_band)
        save_marginals(ztf_name, filter_band)
    end
end

function get_multiband_variation(ztf_folder::String, band1::String, band2::String)
    ENV["GKSwstype"] = "100"
    gr()
    all_files = readdir(ztf_folder, join=true)
    all_prefixes = Set([chop(fn, tail=5) for fn in all_files])
    differences = Dict()
    ct = 0
    for prefix in all_prefixes
        if ct > 100
            break
        end
        try
            file1 = prefix * band1 * ".jls"
            #print(file1)
            file2 = prefix * band2 * ".jls"
            #print(file2)
            trace1 = MCMCChains.read(file1, Chains)
            trace2 = MCMCChains.read(file2, Chains)
            num_chains = length(trace1[1, :A, :])
            for i in 1:num_chains
                A, beta, gamma, t_0, tau_rise, tau_fall, extra_sigma = median(Array(trace1[end-1000:end,:,i]), dims=1)
                single_chain_1 = (;A, beta, gamma, t_0, tau_rise, tau_fall, extra_sigma)
                A, beta, gamma, t_0, tau_rise, tau_fall, extra_sigma = median(Array(trace2[end-1000:end,:,i]), dims=1)
                single_chain_2 = (;A, beta, gamma, t_0, tau_rise, tau_fall, extra_sigma)
                for param in keys(single_chain_1)
                    if abs(single_chain_2[param] / single_chain_1[param]) > 5.
                        continue
                    elseif abs(single_chain_1[param] / single_chain_2[param]) > 5.
                        continue
                    elseif param in keys(differences)
                        push!(differences[param], single_chain_2[param] / single_chain_1[param])
                    else
                        differences[param] = [single_chain_2[param] / single_chain_1[param]]
                    end
                end
            end
            ct += 1
        catch e
            continue
        end
    end
    for param in keys(differences)
        stddev = std(differences[param])
        hist = fit(Histogram, differences[param], nbins=100)
        bin_edges = collect(minimum(hist.edges))
        bin_centers = [(bin_edges[i] + bin_edges[i+1])/2. for i in 1:length(bin_edges)-1]
        bar(bin_centers, hist.weights)
        #xticks!(0:length(hist.weights), hist.edges[1])

        #normal distribution
        model(x, p) = p[3] ./ (p[2] * √2π) .* exp.( -1 .* (x .- p[1]).^2 ./ 2p[2]^2 )
        if param == :t_0
            norm_fit = curve_fit(model, bin_centers, hist.weights, [1., 0.001, 0.5])
        else
            norm_fit = curve_fit(model, bin_centers, hist.weights, [1., 0.5, 200.])
        end
        println(param)
        println(norm_fit.param)
        plot!(bin_centers, model(bin_centers, norm_fit.param), linewidth=3, thickness_scaling = 1)
        savefig("multiband_variation_" * string(param) * ".png")
    end
    
end

function get_multiband_variation2(ztf_folder::String)
    ENV["GKSwstype"] = "100"
    gr()
    all_files = readdir(ztf_folder, join=true)
    #println(all_files)
    #all_prefixes = Set([chop(fn, tail=5) for fn in all_files])
    differences = Dict()
    ct = 0
    for file1 in all_files
        if ct > 100
            break
        end
        #try
        #file1 = prefix * band1 * ".jls"
        #print(file1)
        #file2 = prefix * band2 * ".jls"
        #print(file2)
        trace1 = MCMCChains.read(file1, Chains)
        num_chains = length(trace1[1, :A, :])
        for i in 1:num_chains
            A, beta, gamma, t_0, tau_rise, tau_fall, extra_sigma, A_green, beta_green, gamma_green, t_0_green, tau_rise_green, tau_fall_green = median(Array(trace1[end-1000:end,:,i]), dims=1)
            println(A_green)
            single_chain_1 = (;A_green, beta_green, gamma_green, t_0_green, tau_rise_green, tau_fall_green)
            print(keys(single_chain_1))
            for param in keys(single_chain_1)
                if param in keys(differences)
                    push!(differences[param], single_chain_1[param])
                else
                    differences[param] = [single_chain_1[param]]
                end
            end
        end
        ct += 1
        #catch e
        #    continue
        #end
    end
    for param in keys(differences)
        println(param)
        stddev = std(differences[param])
        hist = fit(Histogram, differences[param], nbins=100)
        bin_edges = collect(minimum(hist.edges))
        bin_centers = [(bin_edges[i] + bin_edges[i+1])/2. for i in 1:length(bin_edges)-1]
        bar(bin_centers, hist.weights)
        #xticks!(0:length(hist.weights), hist.edges[1])

        #normal distribution
        model(x, p) = p[3] ./ (p[2] * √2π) .* exp.( -1 .* (x .- p[1]).^2 ./ 2p[2]^2 )
        if param == :t_0
            norm_fit = curve_fit(model, bin_centers, hist.weights, [1., 0.001, 0.5])
        else
            norm_fit = curve_fit(model, bin_centers, hist.weights, [1., 0.5, 200.])
        end
        println(param)
        println(norm_fit.param)
        plot!(bin_centers, model(bin_centers, norm_fit.param), linewidth=3, thickness_scaling = 1)
        savefig("multiband_variation_" * string(param) * ".png")
    end
    
end

#get_multiband_variation("../../ZTF_fits", "r", "g")
get_multiband_variation2("../../ZTF_fits_multiband")
#plot_fits("ZTF20abwynwj")
