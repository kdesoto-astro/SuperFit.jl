import os
import argparse
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde, median_absolute_deviation
from astropy.table import Table
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
# Default fitting parameters
PHASE_MIN = 58058.
PHASE_MAX = 59528.
ITERATIONS = 10000
TUNING = 25000
WALKERS = 25
CORES = 1
PARAMNAMES = ['Amplitude', 'Plateau Slope (d$^{-1}$)', 'Plateau Duration (d)',
              'Reference Epoch (d)', 'Rise Time (d)', 'Fall Time (d)']


def flux_model(t, A, beta, gamma, t_0, tau_rise, tau_fall):
    """
    Calculate the flux given amplitude, plateau slope, plateau duration, reference epoch, rise time, and fall time using
    theano.switch. Parameters.type = TensorType(float64, scalar).

    Parameters
    ----------
    t : 1-D numpy array
        Time.
    A : TensorVariable
        Amplitude of the light curve.
    beta : TensorVariable
        Light curve slope during the plateau, normalized by the amplitude.
    gamma : TensorVariable
        The duration of the plateau after the light curve peaks.
    t_0 : TransformedRV
        Reference epoch.
    tau_rise : TensorVariable
        Exponential rise time to peak.
    tau_fall : TensorVariable
        Exponential decay time after the plateau ends.

    Returns
    -------
    flux_model : symbolic Tensor
        The predicted flux from the given model.

    """
    phase = t - t_0
    flux_model = A / (1. + tt.exp(-phase / tau_rise)) * \
        tt.switch(phase < gamma, 1. - beta * phase, (1. - beta * gamma) * tt.exp((gamma - phase) / tau_fall))
    return flux_model


class LogUniform(pm.distributions.continuous.BoundedContinuous):
    R"""
    Continuous log-uniform log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid lower, upper) = \frac{1}{[\log(upper)-\log(lower)]x}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(1., 300., 500)
        ls = [3., 150.]
        us = [100., 250.]
        for l, u in zip(ls, us):
            y = np.zeros(500)
            inside = (x<u) & (x>l)
            y[inside] = 1. / ((np.log(u) - np.log(l)) * x[inside])
            plt.plot(x, y, label='lower = {}, upper = {}'.format(l, u))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  =====================================
    Support   :math:`x \in [lower, upper]`
    Mean      :math:`\dfrac{upper - lower}{\log(upper) - \log(lower)}`
    ========  =====================================

    Parameters
    ----------
    lower : float
        Lower limit.
    upper : float
        Upper limit.
    """

    def __init__(self, lower=1., upper=np.e, *args, **kwargs):
        if lower <= 0. or upper <= 0.:
            raise ValueError('LogUniform bounds must be positive')
        log_lower = tt.log(lower)
        log_upper = tt.log(upper)
        self.logdist = pm.Uniform.dist(lower=log_lower, upper=log_upper)
        self.median = tt.exp(self.logdist.median)
        self.mean = (upper - lower) / (log_upper - log_lower)

        super().__init__(lower=lower, upper=upper, *args, **kwargs)

    def random(self, point=None, size=None):
        """
        Draw random values from LogUniform distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        return tt.exp(self.logdist.random(point=point, size=size))

    def logp(self, value):
        """
        Calculate log-probability of LogUniform distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        log_value = tt.log(value)
        return self.logdist.logp(log_value) - log_value


def setup_model1(obs_time, obs_flux, obs_unc):
    """
    Set up the PyMC3 model object, which contains the priors and the likelihood.

    Parameters
    ----------
    obs : astropy.table.Table
        Astropy table containing the light curve data.
    max_flux : float, optional
        The maximum flux observed in any filter. The amplitude prior is 100 * `max_flux`. If None, the maximum flux in
        the input table is used, even though it does not contain all the filters.

    Returns
    -------
    model : pymc3.Model
        PyMC3 model object for the input data. Use this to run the MCMC.
    """
    max_flux = obs_flux.max()
    print("GETTING MAX FLUX")
    with pm.Model() as model:
        A = LogUniform(name=PARAMNAMES[0], lower=1., upper=100. * max_flux)
        beta = pm.Uniform(name=PARAMNAMES[1], lower=0., upper=0.01)
        BoundedNormal = pm.Bound(pm.Normal, lower=0.)
        gamma = pm.Mixture(name=PARAMNAMES[2], w=tt.constant([2., 1.]) / 3., testval=1.,
                           comp_dists=[BoundedNormal.dist(mu=5., sigma=5.), BoundedNormal.dist(mu=60., sigma=30.)])
        t_0 = pm.Uniform(name=PARAMNAMES[3], lower=PHASE_MIN, upper=PHASE_MAX)
        tau_rise = pm.Uniform(name=PARAMNAMES[4], lower=0.01, upper=50.)
        tau_fall = pm.Uniform(name=PARAMNAMES[5], lower=1., upper=300.)
        extra_sigma = pm.HalfNormal(name='Intrinsic Scatter', sigma=1.)
        parameters = [A, beta, gamma, t_0, tau_rise, tau_fall]

        exp_flux = flux_model(obs_time, *parameters)
        sigma = tt.sqrt(tt.pow(extra_sigma, 2.) + tt.pow(obs_unc, 2.))
        pm.Normal(name='Flux_Posterior', mu=exp_flux, sigma=sigma, observed=obs_flux)
    print("FINISHED SETTING UP MODEL")
    return model, parameters

def sample_or_load_trace(model, trace_file, force=True, iterations=ITERATIONS, walkers=WALKERS, tuning=TUNING,
                         cores=CORES):
    """
    Run a Metropolis Hastings MCMC for the given model with a certain number iterations, burn in (tuning), and walkers.

    If the MCMC has already been run, read and return the existing trace (unless `force=True`).

    Parameters
    ----------
    model : pymc3.Model
        PyMC3 model object for the input data.
    trace_file : str
        Path where the trace will be stored. If this path exists, load the trace from there instead.
    force : bool, optional
        Resample the model even if `trace_file` already exists.
    iterations : int, optional
        The number of iterations after tuning.
    walkers : int, optional
        The number of cores and walkers used.
    tuning : int, optional
        The number of iterations used for tuning.
    cores : int, optional
        The number of walkers to run in parallel.

    Returns
    -------
    trace : pymc3.MultiTrace
        The PyMC3 trace object for the MCMC run.
    """
    basename = os.path.basename(trace_file)
    if not os.path.exists(trace_file) or force:
        logging.info(f'Starting fit for {basename}')
        with model:
            trace = pm.sample(iterations, tune=tuning, cores=cores, chains=walkers, step=pm.Metropolis())
        pm.save_trace(trace, trace_file, overwrite=True)
    else:
        trace = pm.load_trace(trace_file)
        logging.info(f'Loaded trace from {trace_file}')
    return trace
