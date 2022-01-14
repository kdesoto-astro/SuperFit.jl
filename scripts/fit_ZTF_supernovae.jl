import Pkg
Pkg.activate("..")
Pkg.instantiate()
Pkg.precompile()

using Distributed


machinefilename = ENV["PBS_NODEFILE"]
println(ENV["PBS_NUM_NODES"])
println(ENV["PBS_NUM_PPN"])
machinespecs = readlines( machinefilename )
threads_per_process = parse(Int64, ENV["PBS_NUM_PPN"])
addprocs(machinespecs[1:threads_per_process:end], exeflags="-t " * string(threads_per_process))

@everywhere import Pkg
@everywhere Pkg.UPDATED_REGISTRY_THIS_SESSION[] = true 
@everywhere Pkg.activate("..")
@everywhere using Dates
println(string("beginning imports", now()))
@everywhere using SuperFit
@everywhere import Turing
@everywhere using Formatting, ArgParse
println(string("finished importing packages", now()))

println("Number of workers: " * string(nworkers()))
println("Number of threads: " * string(Threads.nthreads()))
println("Number of procs: " * string(nprocs()))
open("/proc/$(getpid())/statm") do io
    println("Current pmem: " * split(read(io, String))[1])
end

@everywhere function alg_name_to_algorithm(name)
    if name == "NUTS"
        return Turing.NUTS()
    elseif name == "MH"
        return Turing.MH()
    else
        throw(ArgumentError("Sampling algorithm not supported. Please enter 'NUTS' or 'MH'"))
    end
end

@everywhere function mcmc_parallel(filename, parsed_args)
    """
    Fit the model to the observed light curve. Then combine the posteriors for each filter and use that as the new prior
    for a second iteration of fitting.
    Parameters
    ----------
    light_curve : astropy.table.Table
        Astropy table containing the observed light curve.
    outfile : str
        Path where the trace will be stored. This should include a blank field ({{}}) that will be replaced with the 
        iteration number and filter name. Diagnostic plots will also be saved according to this pattern.
    filters : str, optional
        Light curve filters to fit. Default: 'r' and 'g'
    force : bool, optional
        Redo the fit even if results are already stored in `outfile`. Default: False.
    do_diagnostics : bool, optional
        Produce and save some diagnostic plots. Default: True.
    iterations : int, optional
        The number of iterations between convergence checks.
    walkers : int, optional
        The number of independent chains used.
    Returns
    -------
    traces1, traces2 : dict
        Dictionaries of the PyMC3 trace objects for each filter for the first and second fitting iterations.
    parameters : list
        List of Theano variables in the PyMC3 model.
    """
    println(string("into helper function", now()))
    println("STARTING PIPELINE")
    basename = split(Base.Filesystem.basename(filename), ".")[1]
    outfile = joinpath(parsed_args["output-dir"], string(basename, "{}", ".jls"))
    light_curve = SuperFit.read_light_curve(filename)
    println(string("read lightcurve", now()))
    #if args.zmin is not None and light_curve.meta['REDSHIFT'] <= args.zmin:
    #    raise ValueError(f'Skipping file with redshift {light_curve.meta["REDSHIFT"]}: {filename}')
    t = SuperFit.select_event_data(light_curve)
    #println(string("select event data", now()))
    #traces = Dict()
    filters = parsed_args["filters"]
    algorithm = alg_name_to_algorithm(parsed_args["algorithm"])
    iterations = parsed_args["iterations"]
    walkers = parsed_args["walkers"]
    #TODO: have better filter list checl
    for fltr in filters
        println(string("entered filter loop", now()))
        println(format("STARTING FILTER {}", fltr))
        obs_mags = filter(row -> row.ZTF_filter == fltr, t)
        obs_time, obs_flux, obs_unc = SuperFit.convert_mags_to_flux(obs_mags, SuperFit.ZEROPOINT_MAG)
        println(string("converted mags to flux", now()))
        model = SuperFit.setup_model(obs_time, obs_flux, obs_unc)
        println(string("setup model", now()))
        outfile_fltr = format(outfile, string("_",  fltr))
        converged = SuperFit.sample_or_load_trace(
            model,
            outfile_fltr,
            force=parsed_args["force"],
            algorithm=algorithm,
            iterations=iterations,
            walkers=walkers,
            min_iters=parsed_args["min-iters"],
            max_iters=parsed_args["max-iters"]
        )
        if !converged
            cp(filename, 
                joinpath(parsed_args["unconverged-dir"], string(basename, ".jls")),
                force=true
            )
        end
        #println(string("finished sampling", now()))
        #traces[fltr] = trace
        """
        if do_diagnostics:
            diagnostics(obs, trace1, parameters1, outfile1)
        if do_diagnostics
        end
        """
    end

    """
    TODO: add diagnostics
    if do_diagnostics:
        plot_priors(x_priors, y_priors, old_posteriors, parameters1, outfile.format('_priors.pdf'))
    """
    
    #return traces
    #return traces1, traces2, parameters2
end

function main()
    s = ArgParseSettings()
    @add_arg_table s begin
        "filenames"
            help = "Input light curve data file(s)" 
            nargs = '+'
            arg_type = String
            action = :store_arg
        "--filters"
            help = "Subset of filters to fit"
            arg_type = Vector{String}
            default = SuperFit.FILTERS
            action = :store_arg
        "--iterations"
            help = "Number of steps after burn-in"
            arg_type = Int64
            default = SuperFit.ITERATIONS
            action = :store_arg
        "--walkers"
            help = "Number of walkers"
            arg_type = Int64
            default = SuperFit.WALKERS
            action = :store_arg
        "--unconverged-dir"
            help = "Where unconverged lightcurves are copied to"
            arg_type = String
            default = "unconverged"
            action = :store_arg
        "--min-iters"
            help = "Minimum iterations for convergence"
            arg_type = Int64
            default = 5000
            action = :store_arg
        "--max-iters"
            help = "Maximum iterations for convergence"
            arg_type = Int64
            default = 50000
            action = :store_arg
        "--output-dir"
            help="Path in which to save the trace data"
            arg_type = String
            default = "."
            action = :store_arg
        "--zmin"
            help = "TO BE ADDED"
            action = :store_arg
        "-f", "--force"
            action = :store_true
            help="Redo the fit even if the trace is already saved"
        "--no-plots"
            action = :store_false
            dest_name = "plots"
            help = "Don't save the diagnostic plots"
        "--algorithm"
            help="Sampling algorithm to use. Options currently supported are 'NUTS' or 'MH'.
                See Turing.jl documentation for more information."
            nargs = '?'
            arg_type = String
            default = "NUTS"
            action = :store_arg
    end
    parsed_args = parse_args(ARGS, s)
    #println(string("parsed args", now()))
    #pdf = PdfPages('lc_fits.pdf', keep_empty=False)
    runtime = @elapsed @sync pmap(x -> mcmc_parallel(x, parsed_args), parsed_args["filenames"])
    println(runtime)
    timing_file = "multinode_timing_comparisons.txt"
    open(timing_file, "a+") do tf
        write(tf, string(ENV["PBS_NUM_NODES"],",",ENV["PBS_NUM_PPN"],",",runtime,"\n") )
    end
        
    """
    if args.plots:
    fig = plot_final_fits(light_curve, traces1, traces2, parameters, outfile.format('.pdf'))
    pdf.savefig(fig)
    plt.close(fig)
    """
    #pdf.close()
end

main()
