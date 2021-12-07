using SuperFit
using Distributed
using CSV, DataFrames
using Turing, MCMCChains
using ReverseDiff
using Plots, DataFrames
using LinearAlgebra, PDMats
using Distributions
using Random
using Formatting, ArgParse
using Optim
import AbstractMCMC
import StatsBase

function generate_and_fit_lightcurve(num_times, sigma, trace_file;
        params=SuperFit.generate_random_params(), force=false, algorithm=NUTS(), iterations=SuperFit.ITERATIONS, walkers=SuperFit.WALKERS)
    times, fluxes_w_noise, noise_arr = SuperFit.generate_lightcurve_from_params(num_times, params, sigma)
    model = SuperFit.setup_model(times, fluxes_w_noise, noise_arr)
    SuperFit.sample_or_load_trace(model, trace_file,
        force=force, algorithm=algorithm,
        iterations=iterations, walkers=walkers)
end

function compare_ad_backends(num_points, sigma, num_samples_per_ad)
    trace_file = "../../stored_models/ad_tests.jls"
    
    
    Turing.setadbackend(:reversediff)
    generate_and_fit_lightcurve(num_points, sigma, trace_file, force=true) #run once to fix timing
    times = []
    for i in 1:num_samples_per_ad
        time_elapsed = @elapsed generate_and_fit_lightcurve(num_points, sigma, trace_file, force=true)
        append!(times, time_elapsed)
    end
    println("Reverse diff")
    println(sum(times) / num_samples_per_ad )
    
    Turing.setadbackend(:forwarddiff)
    generate_and_fit_lightcurve(num_points, sigma, trace_file, force=true) #run once to fix timing
    times = []
    for i in 1:num_samples_per_ad
        time_elapsed = @elapsed generate_and_fit_lightcurve(num_points, sigma, trace_file, force=true)
        append!(times, time_elapsed)
    end
    println("Forward diff")
    println(sum(times) / num_samples_per_ad )
end

compare_ad_backends(50, 0.1, 20)