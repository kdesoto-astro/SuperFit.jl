function choose_init_params(mle)
    println(mle.values.array)
    mle_values = mle.values.array
    offsets = [50., 0.001, 1.0, 10.0, 0.1, 200., 5., 20., 0.5]
    #hessian = ReverseDiff.hessian(mle.f, mle.values.array[:, 1])   
    for i in 1:length(offsets)
        try
            std_err = sqrt(abs(StatsBase.informationmatrix(mle)[i]))
            if std_err > 0
                offsets[i] = 3*std_err #3 stddev away
            end
        catch
            println(string("Using manual offset for ", i))
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
        trace_file;
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
    println(Threads.nthreads())
    if !ispath(trace_file) || force
        #logging.info(f'Starting fit for {basename}')
        rhat_plot_arr = zeros(Float64, 1, length(DynamicPPL.syms(DynamicPPL.VarInfo(model))))
        #mle_estimate = optimize(model, MAP(), Optim.Options(iterations=200_000, allow_f_increases=true))
        #println("MLE",mle_estimate.values.array)
        iteration_interval = iterations
        traces = Array{MCMCChains.Chains}(undef, walkers)
        @suppress_err begin
            Threads.@threads for i in 1:walkers
                traces[i] = Turing.sample(model,
                    algorithm,
                    iteration_interval,
                    progress=false,
                    chain_type = MCMCChains.Chains,
                    save_state=true,
                    discard_initial=0)
                    #init_theta = choose_init_params(mle_estimate)
                #)
            end
        end
        while !check_rhat(traces, rhat_plot_arr)
            traces = SuperFit.update_traces(traces, model, algorithm, iteration_interval, walkers)
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
    @suppress_err begin
        Threads.@threads for i in 1:num_walkers
            temp_trace = Turing.sample(model,
                    algorithm,
                    num_iterations,
                    progress=false,
                    resume_from = old_traces[i],
                    chain_type = MCMCChains.Chains,
                    save_state=true,
                    discard_initial=0
            )
            temp_trace_renumbered = MCMCChains.setrange(temp_trace, start_iter:(start_iter+num_iterations-1))
            new_traces[i] = AbstractMCMC.cat(old_traces[i], temp_trace_renumbered, dims=1)
            #CSV.write("cat_chain.csv", DataFrame(new_traces[i]))
        end
        #println(new_traces[i].info)
    end
    return new_traces
end