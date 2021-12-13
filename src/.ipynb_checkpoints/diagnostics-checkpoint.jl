function check_all_diagnostics(traces)
    
    @assert length(traces) > 1 "Must have multiple independent chains to perform diagnostics."
    @assert length(Array(traces[1])) > 0 "Cannot perform diagnostics on empty traces."
    
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

