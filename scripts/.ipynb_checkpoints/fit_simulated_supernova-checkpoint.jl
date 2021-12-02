include("../src/SuperFit.jl")
using .SuperFit

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

const PHASE_MIN = 58058.
const PHASE_MAX = 59528.
const ITERATIONS = 5000
const WALKERS = 6
const FILTERS = ["r", "g"]
const ZEROPOINT_MAG = 22.

function generate_and_fit_lightcurve(num_times, sigma, trace_file;
        params=generate_random_params(), force=false, algorithm=NUTS(), iterations=ITERATIONS, walkers=WALKERS)
    println(num_times)
    times, fluxes_w_noise, noise_arr = SuperFit.generate_lightcurve_from_params(num_times, params, sigma)
    model = SuperFit.setup_model(times, fluxes_w_noise, noise_arr)
    SuperFit.sample_or_load_trace(model, trace_file,
        force=force, algorithm=algorithm,
        iterations=iterations, walkers=walkers)
end

function alg_name_to_algorithm(name)
    if name == "NUTS"
        return NUTS()
    elseif name == "MH"
        return MH()
    else
        throw(ArgumentError("Sampling algorithm not supported. Please enter 'NUTS' or 'MH'"))
    end
end

function main()
    s = ArgParseSettings()
    @add_arg_table s begin
        "-n", "--npoints"
            help = "Number of points in generated lightcurve" 
            required = true
            arg_type = Int64
            action = :store_arg
        "--sigma"
            help = "Standard deviation of lightcurve noise"
            required = true
            arg_type = Float64
            action = :store_arg
        "--output_file"
            help = "Where to store trace output"
            required = true
            arg_type = String
            action = :store_arg
        "--amplitude", "-A"
            help = "Lightcurve amplitude"
            nargs = '?'
            arg_type = Float64
            action = :store_arg
        "--beta"
            help = "Plateau slope"
            nargs = '?'
            arg_type = Float64
            action = :store_arg
        "--gamma_1"
            help = "Plateau duration 1"
            nargs = '?'
            arg_type = Float64
            action = :store_arg
        "--gamma_2"
            help = "Plateau duration 2"
            nargs = '?'
            arg_type = Float64
            action = :store_arg
        "--gamma_switch"
            help = "Between 0 and 1, > 2/3 signifies Type IIn"
            nargs = '?'
            arg_type = Float64
            action = :store_arg
        "--t_0"
            help = "Peak MJD"
            nargs = '?'
            arg_type = Float64
            action = :store_arg
        "--tau_rise"
            help = "Rise slope"
            nargs = '?'
            arg_type = Float64
            action = :store_arg
        "--tau_fall"
            help = "Fall slope"
            nargs = '?'
            arg_type = Float64
            action = :store_arg
        "--walkers"
            help = "Number of independently sampled chains"
            nargs = '?'
            arg_type = Int64
            default = WALKERS
            action = :store_arg
        "--iterations"
            help = "Number of sampler steps between convergence checks."
            nargs = '?'
            arg_type = Int64
            default = ITERATIONS
            action = :store_arg
        "--algorithm"
            help="Sampling algorithm to use. Options currently supported are 'NUTS' or 'MH'.
                See Turing.jl documentation for more information."
            nargs = '?'
            arg_type = String
            default = "NUTS"
            action = :store_arg
        "-f", "--force"
            action = :store_true
            help="Redo the fit even if the trace is already saved"
    end
    parsed_args = parse_args(ARGS, s)
    println(parsed_args)
    #TODO: figure out better way to handle partial parameter entries
    """
    try
        params = (;parsed_args["amplitude"],
            parsed_args["beta"], 
            parsed_args["gamma_1"],
            parsed_args["gamma_2"],
            parsed_args["gamma_switch"],
            parsed_args["t_0"],
            parsed_args["tau_rise"],
            parsed_args["tau_fall"]
        )
        if (parsed_args["A"] == missing ||
        parsed_args["beta"] == nothing ||
        parsed_args["gamma_1"] == nothing ||
        parsed_args["gamma_2"] == nothing ||
        parsed_args["gamma_switch"] == nothing ||
        parsed_args["t_0"] == nothing ||
        parsed_args["tau_rise"] == nothing ||
        parsed_args["tau_fall"] == nothing)
    catch
    """
    params = SuperFit.generate_random_params()
    
    generate_and_fit_lightcurve(
        parsed_args["npoints"],
        parsed_args["sigma"],
        parsed_args["output_file"],
        params=params,
        force=parsed_args["force"],
        algorithm=alg_name_to_algorithm(parsed_args["algorithm"]),
        iterations=parsed_args["iterations"],
        walkers=parsed_args["walkers"])
        
end

main()



