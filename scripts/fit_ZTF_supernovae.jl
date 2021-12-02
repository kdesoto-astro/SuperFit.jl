include("../src/SuperFit.jl")
import .SuperFit

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

function mcmc_multithread(light_curve,
        outfile,
        filters=FILTERS,
        force::Bool=False,
        do_diagnostics::Bool=True,
        iterations::Int64=ITERATIONS,
        walkers::Int64=WALKERS)
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
    println("STARTING PIPELINE")
    t = SuperFit.select_event_data(light_curve)
    traces = Dict()
    #TODO: have better filter list checl
    for fltr in filters
        println(format("STARTING FILTER {}", fltr))
        obs_mags = filter(row -> row.ZTF_filter == fltr, t)
        obs_time, obs_flux, obs_unc = SuperFit.convert_mags_to_flux(obs_mags, ZEROPOINT_MAG)
        model = SuperFit.setup_model(obs_time, obs_flux, obs_unc)
        outfile = format(outfile, string("_",  fltr))
        algorithm = NUTS()
        trace = SuperFit.sample_or_load_trace(model, outfile, force=force, algorithm=algorithm, iterations=iterations, walkers=walkers)
        traces[fltr] = trace
        """
        if do_diagnostics:
            diagnostics(obs, trace1, parameters1, outfile1)
        """
        if do_diagnostics
        end
    end

    """
    TODO: add diagnostics
    if do_diagnostics:
        plot_priors(x_priors, y_priors, old_posteriors, parameters1, outfile.format('_priors.pdf'))
    """
    
    return traces
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
            default = FILTERS
            action = :store_arg
        "--iterations"
            help = "Number of steps after burn-in"
            arg_type = Int64
            default = ITERATIONS
            action = :store_arg
        "--walkers"
            help = "Number of walkers"
            arg_type = Int64
            default = WALKERS
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
    end
    parsed_args = parse_args(ARGS, s)
    #pdf = PdfPages('lc_fits.pdf', keep_empty=False)
    for filename in parsed_args["filenames"]
        basename = split(Base.Filesystem.basename(filename), ".")[1]
        outfile = joinpath(parsed_args["output-dir"], string(basename, "{}", ".jls"))
        light_curve = SuperFit.read_light_curve(filename)
        #if args.zmin is not None and light_curve.meta['REDSHIFT'] <= args.zmin:
        #    raise ValueError(f'Skipping file with redshift {light_curve.meta["REDSHIFT"]}: {filename}')
        traces = mcmc_multithread(light_curve,
            outfile,
            parsed_args["filters"],
            parsed_args["force"],
            parsed_args["plots"],
            parsed_args["iterations"],
            parsed_args["walkers"])
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
