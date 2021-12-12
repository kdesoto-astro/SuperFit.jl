function flux_map(t, params)
    
    #No assertions because the model will skip illegal parameter choices anyways
    #Don't want to interrupt or force/stop sampling by throwing in assertions
    
    gamma = params.gamma_1 / ( 1. + exp(100. * (params.gamma_switch - 2. / 3.))) 
        + params.gamma_2 / ( 1. + exp(-100. * (params.gamma_switch - 2. / 3.)))
    phase = t - params.t_0
    if phase < gamma
        return params.A / (1. + exp(-phase / params.tau_rise)) * (1. - params.beta * phase)
    end
    return params.A / (1. + exp(-phase / params.tau_rise)) * (1. - params.beta * gamma) * exp((gamma - phase) / params.tau_fall)
end

function setup_model(obs_time, obs_flux, obs_unc; max_flux=missing)
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
        A ~ prior_A(max_flux)
        beta ~ prior_beta
        gamma_1 ~ prior_gamma1
        gamma_2 ~ prior_gamma2
        gamma_switch ~ prior_gammaswitch
        t_0 ~ prior_t0
        tau_rise ~ prior_taurise
        tau_fall ~ prior_taufall
        extra_sigma ~ prior_extrasigma #can be a log normal
        params = (;A, beta, gamma_1, gamma_2, gamma_switch, t_0, tau_rise, tau_fall)
        sig = [convert(eltype(beta), s) for s in sig]
        exp_flux = [convert(eltype(beta), flux_map(time, params)) 
            for time in t]
        sigma = sqrt.(sig .^ 2 .+ extra_sigma ^ 2)
        f ~ MvNormal(exp_flux, sigma)
    end
    posterior = sn_model(obs_time, obs_flux, obs_unc)
    #return posterior, parameters
    return posterior
end
