function flux_map(t, b, params)
    
    #No assertions because the model will skip illegal parameter choices anyways
    #Don't want to interrupt or force/stop sampling by throwing in assertions
    
    gamma = params.gamma
    t0 = params.t_0
    A = params.A
    tau_rise = params.tau_rise
    beta = params.beta
    tau_fall = params.tau_fall
    if b == "g"
        A *= params.A_green
        t *= params.t_0_green
        tau_rise *= params.tau_rise_green
        tau_fall *= params.tau_fall_green
        beta *= params.beta_green
        gamma *= params.gamma_green
    end
    
    phase = t - t0
    if phase < gamma
        return params.A / (1. + exp(-phase / params.tau_rise)) * (1. - params.beta * phase)
    end
    return params.A / (1. + exp(-phase / params.tau_rise)) * (1. - params.beta * gamma) * exp((gamma - phase) / params.tau_fall)
end

function setup_model(obs_time, obs_flux, obs_unc, obs_band; max_flux=missing)
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
    @model sn_model(t, f, sig, b) = begin
        A_green ~ prior_g_A
        beta_green ~ prior_g_beta
        gamma_green ~ prior_g_gamma
        t_0_green ~ prior_g_t0
        tau_rise_green ~ prior_g_taurise
        tau_fall_green ~ prior_g_taufall
        extra_sigma_green ~ prior_g_extrasigma
        A ~ prior_A(max_flux)
        beta ~ prior_beta
        gamma ~ prior_gamma
        t_0 ~ prior_t0
        tau_rise ~ prior_tau rise
        tau_fall ~ prior_taufall
        extra_sigma ~ prior_extrasigma
        
        params = (;A_green, beta_green, gamma_green, t_0_green, tau_rise_green, tau_fall_greenA, beta, gamma, t_0, tau_rise, tau_fall,
            )
        sig = [convert(eltype(beta), s) for s in sig]
        exp_flux = [convert(eltype(beta), flux_map(t[i], b[i], params)) 
            for i in 1:len(t)]
        extra_sigma_arr = np.ones(len(sig)) * extra_sigma
        extra_sigma_arr[b == "g"] *= extra_sigma_green
        sigma = sqrt.(sig .^ 2 .+ extra_sigma_arr .^ 2)
        f ~ MvNormal(exp_flux, sigma)
    end
    posterior = sn_model(obs_time, obs_flux, obs_unc, obs_band)
    println(string("create sn model", now()))
    #return posterior, parameters
    return posterior
end
