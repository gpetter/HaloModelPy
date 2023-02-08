import numpy as np
from . import hm_calcs
from . import mcmc
from functools import partial
from scipy.optimize import curve_fit


# a model with a linear power spectrum with a bias corresponding to a constant halo mass
def mass_biased_cf(foo, mass, scales, dndz, hmobject, angular):
    hmobject.set_powspec(log_meff=mass)
    if angular:
        return hmobject.get_binned_ang_cf(dndz, scales)
    else:
        return hmobject.get_binned_spatial_cf(dndz, scales)


# a model with a linear power spectrum with a bias of b
def biased_cf(foo, bias, scales, dndz, hmobject, angular):
    hmobject.set_powspec(bias1=bias)
    if angular:
        return hmobject.get_binned_ang_cf(dndz, scales)
    else:
        return hmobject.get_binned_spatial_cf(dndz, scales)


# fit either a projected or angular correlation function for an effective bias or halo mass
def fit_cf(dndz, cf, angular=False, model='mass'):
    # initialize halo model
    hmobj = hm_calcs.halomodel(zs=dndz[0])
    # if fitting for an effective halo mass
    if model == 'mass':
        partialfun = partial(mass_biased_cf, scales=cf[0], dndz=dndz, hmobject=hmobj, angular=angular)
        popt, pcov = curve_fit(partialfun, np.ones(len(cf[1])), cf[1], sigma=cf[2], absolute_sigma=True,
                               bounds=[11, 14], p0=12.5)
        hmobj.set_powspec(log_meff=popt[0])
        if angular:
            # return the best fit model on a grid for plotting purposes
            modscales = np.logspace(-3, 0., 200)
            bestmodel = (modscales, hmobj.get_ang_cf(dndz, modscales))
        else:
            modscales = np.logspace(-1, 2.3, 200)
            bestmodel = (modscales, hmobj.get_spatial_cf(dndz, radii=modscales))

    # if fitting for an effective bias
    elif model == 'bias':
        partialfun = partial(biased_cf, scales=cf[0], dndz=dndz, hmobject=hmobj, angular=angular)
        popt, pcov = curve_fit(partialfun, np.ones(len(cf[1])), cf[1], sigma=cf[2], absolute_sigma=True,
                               bounds=[0.5, 30], p0=2)
        hmobj.set_powspec(bias1=popt[0])
        if angular:
            modscales = np.logspace(-3, -0, 200)
            bestmodel = (modscales, hmobj.get_ang_cf(dndz, modscales))
        else:
            modscales = np.logspace(-1, 2.3, 200)
            bestmodel = (modscales, hmobj.get_spatial_cf(dndz, radii=modscales))

    # or do full hod modeling with mcmc, not implemented
    else:
        centervals, lowerrs, higherrs = mcmc.sample_cf_space
        
    return popt[0], np.sqrt(pcov)[0][0], bestmodel



