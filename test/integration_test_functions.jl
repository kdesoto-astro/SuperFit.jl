#module FitSNTestFunctions

using SuperFit
using Test
using Distributed
using CSV, DataFrames
using Turing, MCMCChains
using Plots, DataFrames
using LinearAlgebra, PDMats
using Distributions
using Random
using Formatting, ArgParse
using Optim
import AbstractMCMC
import StatsBase
using StatsPlots

function test_simulated_data(num_times, params, noise)
    times, fluxes_w_noise, noise_arr = SuperFit.generate_lightcurve_from_params(
        num_times, params, noise, with_noise=true)
    
    outputfile = "tmp_trace.jls"
    model = SuperFit.setup_model(times, fluxes_w_noise, noise_arr)
    
    SuperFit.sample_or_load_trace(model, outputfile, force=true,
        algorithm=NUTS(), iterations=5000, walkers=3)
    
    ENV["GKSwstype"] = "100"
    trace = MCMCChains.read(outputfile, Chains)
    num_chains = length(trace[1, :A, :])
    test_passed = true
    for i in 1:num_chains
        single_chain = median(Array(trace[:,:,i]), dims=1)
        # checks all params except beta & gamma (can be quite unconstrained)
        
        if (single_chain[1] - params.A) / params.A > 0.1
            test_passed = false
        end
        if (single_chain[end - 3] - params.t_0) / params.t_0 > 0.1
            test_passed = false
        end
        if (single_chain[end - 2] - params.tau_rise) / params.tau_rise > 0.1
            test_passed = false
        end
        if (single_chain[end - 1] - params.tau_fall) / params.tau_fall > 0.1
            test_passed = false
        end
        """
        if (single_chain[5] <= (2. / 3.)) && ((single_chain[3] - gamma_1) / gamma_1 > 0.1)
            test_passed = false
        end
        if (single_chain[5] > (2. / 3. )) && ((single_chain[4] - gamma_2) / gamma_2 > 0.1)
            test_passed = false
        end
        """
    end
    rm(outputfile)
    return test_passed
end

#@test test_simulated_data(50, 100., 0.002, 6., 70., 0.1, 58840., 15., 50., 0.5)
