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

prior_g_A = TruncatedNormal(1.027, 0.152, 0., Inf)
prior_g_beta = TruncatedNormal(1.035, 0.161, 0., Inf)
prior_g_gamma = TruncatedNormal(1.021, 0.278, 0., Inf)
prior_g_t0 = TruncatedNormal(1.0, 0.000365, 0., Inf)
prior_g_taurise = TruncatedNormal(0.953, 0.284, 0., Inf)
prior_g_taufall = TruncatedNormal(0.522, 0.102, 0., Inf)
prior_g_extrasigma = TruncatedNormal(0.895, 0.418, 0., Inf)
    
