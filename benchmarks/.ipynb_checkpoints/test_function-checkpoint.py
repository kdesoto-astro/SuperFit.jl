import pymc3 as pm
from astropy.table import Table
from superphot_fit import setup_model1
from superphot_fit import sample_or_load_trace

def superphot_fit(obs_time, obs_flux, obs_unc, outfile):
    model_phot = setup_model1(obs_time, obs_flux, obs_unc)
    trace_phot = sample_or_load_trace(model_phot, outfile)
    return trace_phot
