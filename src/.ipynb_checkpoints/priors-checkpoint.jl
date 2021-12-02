function prior_A(max_flux::Float64)
    return LogUniform(1.0001, 100. * max_flux)
end

prior_beta = Uniform(0. , 0.01)
prior_gamma1 = TruncatedNormal(5., 5., 0., Inf)
prior_gamma2 = TruncatedNormal(60., 30., 0., Inf)
prior_gammaswitch = Uniform(0., 1.)
prior_t0 = Uniform(PHASE_MIN, PHASE_MAX)
prior_taurise = Uniform(0.01, 50.)
prior_taufall = Uniform(1., 300.)
prior_extrasigma = TruncatedNormal(0., 1., 0.001, Inf)
    
