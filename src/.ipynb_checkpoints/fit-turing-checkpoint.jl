#PARAMNAMES = ['Amplitude', 'Plateau Slope (d$^{-1}$)', 'Plateau Duration (d)',
#              'Reference Epoch (d)', 'Rise Time (d)', 'Fall Time (d)']

function flux_map(phase, A, beta, gamma, tau_rise, tau_fall)
    if phase < gamma
        return A / (1. + exp(-phase / tau_rise)) * (1. - beta * phase)
    end
    return A / (1. + exp(-phase / tau_rise)) * (1. - beta * gamma) * exp((gamma - phase) / tau_fall)
end

function flux_model(t, A, beta, gamma_1, gamma_2, gamma_switch, t_0, tau_rise, tau_fall)
    """
    Calculate the flux (IN MAGNITUDES) given amplitude, plateau slope, plateau duration, reference epoch, rise time, and fall time using
    theano.switch. Parameters.type = TensorType(float64, scalar).
    Parameters
    ----------
    t : 1-D array
        Time.
    A : Constant
        Amplitude of the light curve.
    beta : Constant
        Light curve slope during the plateau, normalized by the amplitude.
    gamma : Constant
        The duration of the plateau after the light curve peaks.
    t_0 : Constant
        Reference epoch.
    tau_rise : Constant
        Exponential rise time to peak.
    tau_fall : Constant
        Exponential decay time after the plateau ends.
    Returns
    -------
    flux_model : 1-D array
        The predicted flux from the given model.
    """
    """
    gamma = gamma_2
    if gamma_switch <= ( 2. / 3. )
        gamma = gamma_1
    end
    """
    gamma = gamma_1 / ( 1. + exp(100. * (gamma_switch - 2. / 3.))) + gamma_2 / ( 1. + exp(-100. * (gamma_switch - 2. / 3.)))
    phase = t .- t_0
    flux_model = [convert(eltype(beta),flux_map(p, A, beta, gamma, tau_rise, tau_fall)) for p in phase]
    return flux_model
end
    
function setup_model1(obs_time, obs_flux, obs_unc, max_flux::Float64=None)
    """
    Set up the Turing model object, which contains the priors and the likelihood.
    Parameters
    ----------
    obs : DataFrame
        Dataframe containing the light curve data.
    max_flux : float, optional
        The maximum flux observed in any filter. The amplitude prior is 100 * `max_flux`. If None, the maximum flux in
        the input table is used, even though it does not contain all the filters.
    Returns
    -------
    posterior : Turing model posterior
        Posterior space for the input data and likelihood functions. Use this to run the MCMC.
    """
    println("SETTING UP MODEL")
    if ismissing(max_flux)
        max_flux = Base.maximum(obs_flux)
    end
    if max_flux <= 0.01
        throw(ArgumentError("The maximum flux is very low. Cannot fit the model."))
    end
    @model sn_model(t, f, sig) = begin
        A ~ LogUniform(1.0001, 100. * max_flux)
        beta ~ Uniform(0. , 0.01)
        gamma_1 ~ TruncatedNormal(5., 5., 0., Inf)
        gamma_2 ~ TruncatedNormal(60., 30., 0., Inf)
        gamma_switch ~ Uniform(0., 1.)
        t_0 ~ Uniform(PHASE_MIN, PHASE_MAX)
        tau_rise ~ Uniform(0.01, 50.)
        tau_fall ~ Uniform(1., 300.)
        extra_sigma ~ TruncatedNormal(0., 1., 0.0, Inf) #can be a log normal
        sig = [convert(eltype(beta), s) for s in sig]
        exp_flux = flux_model(t, A, beta, gamma_1, gamma_2, gamma_switch, t_0, tau_rise, tau_fall)
        sigma = sqrt.(sig .^ 2. .+ extra_sigma ^ 2.)
        f ~ MvNormal(exp_flux, sigma)
    end
    posterior = sn_model(obs_time, obs_flux, obs_unc)
    #return posterior, parameters
    return posterior
end

function check_rhat(traces, rhat_plot_arr)
    samples = AbstractMCMC.chainsstack(traces)
    iteration = range(samples)[end]
    sumstats = DataFrame(summarystats(samples))
    println(iteration, sumstats.rhat)
    rhat_plot_arr = vcat(rhat_plot_arr, transpose(Array{Float64}(sumstats.rhat)))
    rhat_vec = sumstats.rhat
    if iteration > 200000
        return true
    end
    if iteration < 30000
        return false
    end
    for rhat in rhat_vec
        if ismissing(rhat) || rhat > 1.01
            return false
        end
    end
    return true
end

function choose_init_params(mle)
    println(mle.values.array)
    mle_values = mle.values.array
    offsets = [50., 0.0002, 1.0, 10.0, 0.1, 200., 5., 20., 0.5]
    for i in 1:length(offsets)
        try
            offsets[i] = 3*StatsBase.stderror(mle)[i] #3 stddev away 
        catch
            continue
        end
    end
    init_params = []
    for i in 1:length(mle_values)
        random_offset = rand(Uniform(-1*offsets[i], offsets[i]))
        if i == 1
            append!(init_params, max(1.,  mle_values[i] + random_offset))
        elseif i == 2
            append!(init_params, min(max(0., mle_values[i] + random_offset), 0.01))
        elseif i == 8
            append!(init_params, min(max(1., mle_values[i] + random_offset), 300.))
        elseif i == 7
            append!(init_params, min(max(0.01, mle_values[i] + random_offset), 50.))
        elseif i == 5
            append!(init_params, min(max(0. , mle_values[i] + random_offset), 1.))
        else
            append!(init_params, max(0. , mle_values[i] + random_offset))
        end
    end
    println(string("init ", init_params))
    return init_params
end

function sample_or_load_trace(model,
        trace_file,
        force::Bool=False,
        algorithm=NUTS(),
        iterations::Int64=ITERATIONS,
        walkers::Int64=WALKERS)
    """
    Run a type of MCMC for the given model with a certain number iterations per check, and independent walkers.
    If the MCMC has already been run, read and return the existing trace (unless `force=True`).
    Parameters
    ----------
    model : Turing.model
        Turing model object for the input data.
    trace_file : str
        Path where the trace will be stored. If this path exists, load the trace from there instead.
    force : bool, optional
        Resample the model even if `trace_file` already exists.
    iterations : int, optional
        The number of iterations between convergence checks.
    walkers : int, optional
        The number of total independent chains.
    Returns
    -------
    trace : Chains
        A Julia Chains object for the MCMC run.
    """
    #basename = os.path.basename(trace_file)
    println("SAMPLING AND LOADING TRACE")
    if !ispath(trace_file) || force
        #logging.info(f'Starting fit for {basename}')
        rhat_plot_arr = zeros(Float64, 1, length(DynamicPPL.syms(DynamicPPL.VarInfo(model))))
        mle_estimate = optimize(model, MAP(), Optim.Options(iterations=100_000, allow_f_increases=true))
        #println("MLE",mle_estimate.values.array)
        iteration_interval = iterations
        traces = Array{MCMCChains.Chains}(undef, walkers)
        Threads.@threads for i in 1:walkers
            traces[i] = Turing.sample(model,
                algorithm,
                iteration_interval,
                progress=true,
                chain_type = MCMCChains.Chains,
                save_state=true,
                discard_initial=0,
                init_theta = choose_init_params(mle_estimate)
            )
        end
        while !check_rhat(traces, rhat_plot_arr)
            traces = update_traces(traces, model, algorithm, iteration_interval, walkers)
        end
        trace = AbstractMCMC.chainsstack(traces)
        write(trace_file, trace)
        CSV.write("rhat_arr.csv", DataFrame(rhat_plot_arr, :auto))
    else
        trace = read(trace_file, Chains)
        #logging.info(f'Loaded trace from {trace_file}')
    end
    return trace
end

function update_traces(old_traces, model, algorithm, num_iterations, num_walkers)
    new_traces = Array{MCMCChains.Chains}(undef, num_walkers)
    samples = AbstractMCMC.chainsstack(old_traces)
    start_iter = range(samples)[end]+1
    Threads.@threads for i in 1:num_walkers
        temp_trace = Turing.sample(model,
                algorithm,
                num_iterations,
                progress=true,
                resume_from = old_traces[i],
                chain_type = MCMCChains.Chains,
                save_state=true,
                discard_initial=0
        )
        temp_trace_renumbered = MCMCChains.setrange(temp_trace, start_iter:(start_iter+num_iterations-1))
        CSV.write("renum_chain.csv", DataFrame(temp_trace_renumbered))
        new_traces[i] = AbstractMCMC.cat(old_traces[i], temp_trace_renumbered, dims=1)
        CSV.write("cat_chain.csv", DataFrame(new_traces[i]))
        #println(new_traces[i].info)
    end
    return new_traces
end

function select_event_data(t::DataFrame, phase_min::Float64=PHASE_MIN, phase_max::Float64=PHASE_MAX, nsigma=missing)
    """
    Select data only from the period containing the peak flux, with outliers cut.
    Parameters
    ----------
    t : DataFrame
        DataFrame containing the light curve data.
    phase_min, phase_max : float, optional
        Include only points within [`phase_min`, `phase_max`) days of SEARCH_PEAKMJD.
    nsigma : float, optional
        Determines at what value (flux < nsigma * mad_std) to reject outlier data points. Default: no rejection.
    Returns
    -------
    t_event : DataFrame
        Table containing the reduced light curve data from the period containing the peak flux.
    """
    t_event = t
    #t_event = t[(t['PHASE'] >= phase_min) & (t['PHASE'] < phase_max)]
    #TODO: current pipeline assumes event already isolated in file, change this later
    if !ismissing(nsigma)
        t_event = cut_outliers(t_event, nsigma)
    end
    return t_event
end

function read_light_curve(filename)
    """
    Read light curve data from SSV format, and convert important columns to DataFrame
    Parameters
    ----------
    filename : str
        Path to light curve data file.
    Returns
    -------
    df : DataFrame
        DataFrame of light curve data.
    """
    #TODO: this function will change when using DR7 FITS file format, or LSST file formats
    println(filename)
    df = CSV.read(filename, DataFrame, delim=" ")
    println(df)
    return df
end

function cut_outliers(t::DataFrame, nsigma::Float64)
    """
    Make a DataFrame containing only data that is below the cut off threshold.
    Parameters
    ----------
    t : DataFrame
        DataFrame object containing the light curve data.
    nsigma : float
        Determines at what value (flux < nsigma * mad_std) to cut outlier data points.
    Returns
    -------
    t_cut : DataFrame
        Filtered DataFrame object containing only data that is below the cut off threshold.
    """
    madstd = median_absolute_deviation(t.ZTF_PSF)
    t_cut = filter(row -> row.ZTF_PSF < nsigma * madstd, t)
    return t_cut
end

function median_absolute_deviation(mags)
    """
    Calculates the mean absolute deviation:
    MAD = Median(abs(mag - avg mag))
    """
    return Base.median(abs.(mags .- mean(mags)))
end

function convert_mags_to_flux(obs, zp)
    fluxes = Array{Float64}(undef, length(obs.ZTF_PSF))
    flux_unc = Array{Float64}(undef, length(obs.ZTF_PSFerr))
    for i in 1:length(obs.ZTF_PSF)
        m = obs.ZTF_PSF[i]
        fluxes[i] = 10. ^ (-1. * ( m - zp ) / 2.5)
        flux_unc[i] = log(10.)/2.5 * fluxes[i] * obs.ZTF_PSFerr[i]
    end
    return obs.ZTF_MJD, fluxes, flux_unc
end

function two_iteration_mcmc(light_curve,
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
    t = select_event_data(light_curve)
    traces1 = Dict()
    #TODO: have better filter list checl
    for fltr in filters
        zeropoint_mag = 22.
        println(format("STARTING FILTER {}", fltr))
        obs_mags = filter(row -> row.ZTF_filter == fltr, t)
        obs_time, obs_flux, obs_unc = convert_mags_to_flux(obs_mags, zeropoint_mag)
        model1 = setup_model1(obs_time, obs_flux, obs_unc, Base.maximum(obs_flux))
        outfile1 = format(outfile, string("_1",  fltr))
        algorithm = NUTS()
        trace1 = sample_or_load_trace(model1, outfile1, force, algorithm, iterations, walkers)
        traces1[fltr] = trace1
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
    
    return traces1
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
        outfile = joinpath(parsed_args["output-dir"], string(basename, "{}"))
        light_curve = read_light_curve(filename)
        #if args.zmin is not None and light_curve.meta['REDSHIFT'] <= args.zmin:
        #    raise ValueError(f'Skipping file with redshift {light_curve.meta["REDSHIFT"]}: {filename}')
        traces1 = two_iteration_mcmc(light_curve,
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