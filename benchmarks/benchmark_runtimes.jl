using Distributed
@everywhere import Pkg
@everywhere Pkg.offline(true)
@everywhere Pkg.activate("..")

@everywhere using SuperFit
@everywhere using CSV, DataFrames
@everywhere using Turing, MCMCChains
@everywhere using Plots, DataFrames
@everywhere using LinearAlgebra, PDMats
@everywhere using Distributions
@everywhere using Random
@everywhere using Formatting, ArgParse
@everywhere using Optim
@everywhere import AbstractMCMC
@everywhere import StatsBase

const PHASE_MIN = 58058.
const PHASE_MAX = 59528.
const ITERATIONS = 5000
const WALKERS = 6
const FILTERS = ["r", "g"]
const ZEROPOINT_MAG = 22.

@everywhere function generate_and_fit_lightcurve(num_times, sigma, trace_file;
        params=generate_random_params(), force=false, algorithm=NUTS(), iterations=ITERATIONS, walkers=WALKERS)
    println(num_times)
    times, fluxes_w_noise, noise_arr = SuperFit.generate_lightcurve_from_params(num_times, params, sigma)
    model = SuperFit.setup_model(times, fluxes_w_noise, noise_arr)
    time = @elapsed SuperFit.sample_or_load_trace(model, trace_file,
        force=force, algorithm=algorithm,
        iterations=iterations, walkers=walkers)
    return time
end

@everywhere function alg_name_to_algorithm(name)
    if name == "NUTS"
        return NUTS()
    elseif name == "MH"
        return MH()
    else
        throw(ArgumentError("Sampling algorithm not supported. Please enter 'NUTS' or 'MH'"))
    end
end

@everywhere function do_parallelism(npoints, parsed_args)
    
    params = SuperFit.generate_random_params()
    
    output=joinpath(parsed_args["output_dir"], "sim_" * string(npoints) * ".jls")
    timing_file=joinpath(parsed_args["timing_dir"], "sim_" * string(npoints) * ".txt")
    open(timing_file, "w+") do tf
        write(tf, "")
    end
    open(timing_file, "a+") do tf
        write(tf, "Number of points: " * string(npoints))
        write(tf, "\nFraction uncertainty: " * string(parsed_args["sigma"]))
        write(tf, "\n\nRuntimes:\n")
    end
    runtimes = zeros()
    for i in 1:parsed_args["nsimulations"]
        runtime = generate_and_fit_lightcurve(
            npoints,
            parsed_args["sigma"],
            output,
            params=params,
            force=parsed_args["force"],
            algorithm=alg_name_to_algorithm(parsed_args["algorithm"]),
            iterations=parsed_args["iterations"],
            walkers=parsed_args["walkers"])
        open(timing_file, "a+") do tf
            write(tf, string(runtime))
            write(tf, "\n")
        end
        append!(runtimes, runtime)
    end
    open(timing_file, "a+") do tf
        write(tf, "\nMedian of runtimes:\n" * string(median(runtimes)))
    end
end

@everywhere function main()
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
        "--output_dir"
            help = "Where to store trace output"
            required = true
            arg_type = String
            action = :store_arg
        "--timing_dir"
            help = "Where to store time analysis"
            required = true
            arg_type = String
            action = :store_arg
        "--nsimulations"
            help = "Number of simulations to take median of"
            nargs = '?'
            arg_type = Int64
            action = :store_arg
            default = 7
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
    parsed_args = parse_args(ARGS, s)
    println(parsed_args)
    
    machinefilename = ENV["PBS_NODEFILE"]
    machinespecs = readlines( machinefilename )
    threads_per_process = 2
    num_processors = length(machinespecs[1:threads_per_process:end]) - 1
    addprocs(num_processors, exeflags="-t 2")
    println("Number of workers: " * string(nworkers()))
    println("Number of threads: " * string(Threads.nthreads()))
    println("Number of procs: " * string(nprocs()))
    npoint_arr = 10:10:100
    println(npoint_arr)
    pmap(x -> do_parallelism(x, parsed_args), collect(npoint_arr))
end

main()



