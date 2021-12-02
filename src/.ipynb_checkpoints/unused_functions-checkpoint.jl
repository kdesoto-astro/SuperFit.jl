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

function check_all_diagnostics(traces)
    samples = AbstractMCMC.chainsstack(traces)
    println("Geweke")
    geweke = MCMCChains.gewekediag(traces[1], first=0.1, last=0.5)
    write("geweke.txt", geweke)
    println("Heidelberger-Welch")
    hw = MCMCChains.heideldiag(traces[1], alpha=0.05, eps=0.1)
    CSV.write("heidel.csv", DataFrame(hw))
    println("Gelman-Rubin")
    gr = MCMCChains.gelmandiag_multivariate(samples)
    CSV.write("gelman.csv", DataFrame(gr))
end

