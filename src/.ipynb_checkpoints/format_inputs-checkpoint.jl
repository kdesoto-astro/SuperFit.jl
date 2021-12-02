function generate_random_params()
    A = rand(prior_A(10.))
    beta = rand(prior_beta)
    gamma_1 = rand(prior_gamma1)
    gamma_2 = rand(prior_gamma2)
    gamma_switch = rand(prior_gammaswitch)
    t_0 = rand(prior_t0)
    tau_rise = rand(prior_taurise)
    tau_fall = rand(prior_taufall)
    return (;A, beta, gamma_1, gamma_2, gamma_switch, t_0, tau_rise, tau_fall)
end

function generate_lightcurve_from_params(num_times, params, noise_frac; with_noise=true)
    fluxes_w_noise = Array{Float64}(undef, num_times)
    times = Array{Float64}(undef, num_times)
    noise_arr = Array{Float64}(undef, num_times)
    noise = noise_frac * params.A
    for i in 1:num_times
        time = rand(Uniform(-100. + params.t_0, 200. + params.t_0))
        times[i] = time
        flux_generated = flux_map(time, params)
        added_noise = rand(Normal(0., noise))
        if with_noise
            noise_arr[i] = noise
            fluxes_w_noise[i] = max(flux_generated + added_noise, 0.)
        else
            noise_arr[i] = 0.001
            fluxes_w_noise[i] = flux_generated
        end
    end
    sort_idx = sortperm(times)
    return times[sort_idx], fluxes_w_noise[sort_idx], noise_arr[sort_idx]
end

function select_event_data(t::DataFrame, phase_min::Float64=PHASE_MIN, phase_max::Float64=PHASE_MAX, nsigma=missing)
    """
    Select data only from the period containing the peak flux, with outliers cut.
    Parameters
    ----------
    t : DataFrame
        DataFrame containing the light curve data.
    phase_min, phase_max : float, optional
        Include only points within [`phase_min`, `phase_max`) days of SEARCH_PEAKMJD.
    nsigma : float, optional
        Determines at what value (flux < nsigma * mad_std) to reject outlier data points. Default: no rejection.
    Returns
    -------
    t_event : DataFrame
        Table containing the reduced light curve data from the period containing the peak flux.
    """
    t_event = t
    #t_event = t[(t['PHASE'] >= phase_min) & (t['PHASE'] < phase_max)]
    #TODO: current pipeline assumes event already isolated in file, change this later
    if !ismissing(nsigma)
        t_event = cut_outliers(t_event, nsigma)
    end
    return t_event
end

function read_light_curve(filename)
    """
    Read light curve data from SSV format, and convert important columns to DataFrame
    Parameters
    ----------
    filename : str
        Path to light curve data file.
    Returns
    -------
    df : DataFrame
        DataFrame of light curve data.
    """
    #TODO: this function will change when using DR7 FITS file format, or LSST file formats
    println(filename)
    df = CSV.read(filename, DataFrame, delim=" ")
    println(df)
    return df
end

function cut_outliers(t::DataFrame, nsigma::Float64)
    """
    Make a DataFrame containing only data that is below the cut off threshold.
    Parameters
    ----------
    t : DataFrame
        DataFrame object containing the light curve data.
    nsigma : float
        Determines at what value (flux < nsigma * mad_std) to cut outlier data points.
    Returns
    -------
    t_cut : DataFrame
        Filtered DataFrame object containing only data that is below the cut off threshold.
    """
    madstd = median_absolute_deviation(t.ZTF_PSF)
    t_cut = filter(row -> row.ZTF_PSF < nsigma * madstd, t)
    return t_cut
end

function median_absolute_deviation(mags)
    """
    Calculates the mean absolute deviation:
    MAD = Median(abs(mag - avg mag))
    """
    return Base.median(abs.(mags .- mean(mags)))
end

function convert_mags_to_flux(obs, zp)
    fluxes = Array{Float64}(undef, length(obs.ZTF_PSF))
    flux_unc = Array{Float64}(undef, length(obs.ZTF_PSFerr))
    for i in 1:length(obs.ZTF_PSF)
        m = obs.ZTF_PSF[i]
        fluxes[i] = 10. ^ (-1. * ( m - zp ) / 2.5)
        flux_unc[i] = log(10.)/2.5 * fluxes[i] * obs.ZTF_PSFerr[i]
    end
    return obs.ZTF_MJD, fluxes, flux_unc
end

