function choose_init_params(mle)
    #println(mle.values.array)
    mle_values = mle.values.array
    offsets = [0.5*m for m in mle_values]
    #hessian = ReverseDiff.hessian(mle.f, mle.values.array[:, 1])
    
    for i in 1:length(offsets)
        #println(std_err)
        try
            std_err = StatsBase.stderror(mle)[i]
            #std_err = sqrt(abs(StatsBase.informationmatrix(mle)[i]))
            if std_err > 0
                offsets[i] = 3*std_err #3 stddev away
            else
                #println(string("Using manual offset for ", i))
            end
        catch
            #println(string("Using manual offset for ", i))
        end
    end
    init_params = []
    for i in 1:length(mle_values)
        random_offset = rand(Uniform(-1*offsets[i], offsets[i]))
        if i == 1
            append!(init_params, max(1.,  mle_values[i] + random_offset))
        elseif i == 2
            append!(init_params, min(max(0., mle_values[i] + random_offset), 0.01))
        """
        elseif i == 8
            append!(init_params, min(max(1., mle_values[i] + random_offset), 300.))
        elseif i == 7
            append!(init_params, min(max(0.01, mle_values[i] + random_offset), 50.))
        elseif i == 5
            append!(init_params, min(max(0. , mle_values[i] + random_offset), 1.))
        """
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
        walkers::Int64=WALKERS,
        min_iters::Int64=5000,
        max_iters::Int64=50000
    )
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
    global_logger(ConsoleLogger(stderr, Logging.Error))
    
    #Number of iterations and walkers must be positive
    @assert iterations > 0 "Number of iterations between convergence checks must be positive."
    @assert walkers > 1 "Number of chains must be greater than 1 for Gelman-Rubin convergence diagnostic."
    
    println("SAMPLING AND LOADING TRACE")
    #println(string("in sampler function", now()))
    converged = true
    if !ispath(trace_file) || force
        #logging.info(f'Starting fit for {basename}')
        rhat_plot_arr = zeros(Float64, 1, length(DynamicPPL.syms(DynamicPPL.VarInfo(model))))
        iteration_interval = iterations
        traces = Array{MCMCChains.Chains}(undef, walkers)
        try
            #time = @elapsed optimize(model, MAP(), Optim.Options(iterations=00_000, allow_f_increases=true))
            #println(time)
            mle_estimate = optimize(model, MAP(), BFGS(), Optim.Options(iterations=100_000, allow_f_increases=true))
            #println(StatsBase.informationmatrix(mle_estimate))
            println(string("map estimate", now()))
            println("Using MAP estimate")
            Threads.@threads for i in 1:walkers
                traces[i] = Turing.sample(
                    model,
                    algorithm,
                    iteration_interval,
                    progress=false,
                    chain_type=MCMCChains.Chains,
                    save_state=true,
                    discard_initial=0,
                    init_theta=choose_init_params(mle_estimate)
                )
            end
        catch
            Threads.@threads for i in 1:walkers
                traces[i] = Turing.sample(
                    model,
                    algorithm,
                    iteration_interval,
                    progress=false,
                    chain_type=MCMCChains.Chains,
                    save_state=true,
                    discard_initial=0
                )
            end
        end
        trace = AbstractMCMC.chainsstack(traces)
        num_iters = range(trace)[end]
        while (num_iters <= max_iters) && 
            (num_iters < min_iters || !check_rhat(trace, rhat_plot_arr))
            trace = SuperFit.update_traces(trace, model, algorithm, iteration_interval, walkers)
            num_iters = range(trace)[end]
        end
        
        if num_iters > max_iters
            converged = false
        end
        #trace = AbstractMCMC.chainsstack(traces)
        println(length(trace[1,:,1]))
        write(trace_file, trace)
        #println(string("wrote to file", now()))
        CSV.write("rhat_arr.csv", DataFrame(rhat_plot_arr, :auto))
    else
        trace = read(trace_file, Chains)
        #logging.info(f'Loaded trace from {trace_file}')
    end
    
    return converged
end

function update_traces(samples, model, algorithm, num_iterations, num_walkers)
    
    # @assert length(old_traces) == num_walkers "Number of traces must equal number of specified chains."
    @assert num_iterations > 0 "Number of iterations to sample must be positive."
    @assert num_walkers > 0 "Number of chains must be positive."
    
    #new_traces = Array{MCMCChains.Chains}(undef, num_walkers)
    #samples = AbstractMCMC.chainsstack(old_traces)
    #println(string("stacked traces", now()))
    start_iter = range(samples)[end]+1
    temp_trace = Turing.sample(model,
        algorithm,
        MCMCThreads(),
        num_iterations,
        num_walkers,
        progress=false,
        resume_from = samples,
        chain_type = MCMCChains.Chains,
        save_state=true,
        discard_initial=0
    )
    traces_renumbered = MCMCChains.setrange(temp_trace, start_iter:(start_iter+num_iterations-1))
    new_traces = AbstractMCMC.cat(samples, traces_renumbered, dims=1)
    #Check invariants
    #@assert length(old_traces) == length(new_traces) "Number of chains should not change during sampling."
    
    return new_traces
end