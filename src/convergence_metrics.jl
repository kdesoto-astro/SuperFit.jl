function check_rhat(traces, rhat_plot_arr)
    samples = AbstractMCMC.chainsstack(traces)
    iteration = range(samples)[end]
    sumstats = DataFrame(summarystats(samples))
    println(iteration, sumstats.rhat)
    rhat_plot_arr = vcat(rhat_plot_arr, transpose(Array{Float64}(sumstats.rhat)))
    rhat_vec = sumstats.rhat
    if iteration > 100000
        return true
    end
    """
    if iteration > 200000
        return true
    end
    """
    if iteration < 10000
        return false
    end
    for rhat in rhat_vec
        if ismissing(rhat) || rhat > 1.05
            return false
        end
    end
    return true
end

function check_rhat_temp(traces)
    samples = AbstractMCMC.chainsstack(traces)
    CSV.write("combined_chains.csv", DataFrame(samples))
    iteration = range(samples)[end]
    println(iteration)
    if iteration > 5000
        return true
    end
    return false
end

function calc_rhat_from_array(traces)
    N = length(Array(traces[1]))
    M = length(traces) #number of parallel chains
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
    println(W)
    println(B)
    println(string("Manual r-hat: ", R))
    return R
end
