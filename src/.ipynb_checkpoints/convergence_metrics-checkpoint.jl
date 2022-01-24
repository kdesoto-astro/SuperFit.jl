function check_rhat(samples, rhat_plot_arr)
    
    #@assert length(traces) > 1 "Gelman-Rubin diagnostic only works for > 1 chains."

    rhat_threshold = 1.05
    
    #samples = AbstractMCMC.chainsstack(traces)
    iteration = range(samples)[end]
    
    @assert iteration > 0 "Cannot calculate summary statistics for empty traces."
    
    sumstats = DataFrame(summarystats(samples))
    println(iteration, sumstats.rhat, now())
    
    rhat_plot_arr = vcat(rhat_plot_arr, transpose(Array{Float64}(sumstats.rhat)))
    rhat_vec = sumstats.rhat
    """
    if iteration > max_iteration
        return true
    end
    if iteration < min_iteration
        return false
    end
    """
    for rhat in rhat_vec
        if ismissing(rhat) || rhat > rhat_threshold
            return false
        end
    end
    return true
end

function check_rhat_temp(traces)
    
    min_iterations = 5000
    samples = AbstractMCMC.chainsstack(traces)
    iteration = range(samples)[end]
    #println(iteration)
    if iteration > min_iterations
        return true
    end
    return false
end

function calc_rhat_from_array(traces)
    N = length(Array(traces[1]))
    M = length(traces) #number of parallel chains
    
    @assert M > 1 "Number of traces must be > 1"
    @assert N > 0 "Length of traces must be positive"
    
    W = 0.
    trace_means = []
    for trace in traces
        trace_arr = Array(trace)
        append!(trace_means, mean(trace_arr))
        W += var(trace_arr)
    end
    W = W / M
    overall_mean = mean(trace_means)
    B = (N/(M-1)) * var(trace_means)
    V = (N-1)/N * W + (M+N)/(M*N) * B
    R = sqrt(V / W)
    #println(W)
    #println(B)
    println(string("Manual r-hat: ", R))
    return R
end
