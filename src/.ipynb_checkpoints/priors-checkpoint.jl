function prior_A(max_flux::Float64)
    @assert max_flux > 0 "Maximum flux must be positive"
    return LogUniform(1.0001, 100. * max_flux)
end

prior_beta = Uniform(0., 0.01)
prior_gamma = Truncated(MixtureModel(Normal, [(5.0, 5.0), (60.0, 30.0)], [0.67, 0.33]), 0.0, Inf)
#prior_gamma1 = TruncatedNormal(5., 5., 0., 18.15)
#prior_gamma2 = TruncatedNormal(60., 30., 18.15, Inf)
#prior_gammaswitch = Uniform(0., 0.5)
prior_t0 = Uniform(PHASE_MIN, PHASE_MAX)
prior_taurise = Uniform(0.01, 50.)
prior_taufall = Uniform(1., 300.)
prior_extrasigma = TruncatedNormal(0., 1., 0.001, Inf)
    
