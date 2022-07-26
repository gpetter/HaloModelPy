import numpy as np
from . import hm_calcs
from functools import partial
from scipy.optimize import curve_fit


# functions to call for fits
# take a halo mass or bias factor, return model cross correlation or kappa stack
def mass_biased_xcorr(foo, mass, ells, dndz, hmobject):
	hmobject.set_powspec(log_meff=mass)
	return hmobject.get_binned_c_ell_kg(dndz=dndz, ls=ells)


def mass_biased_stack(foo, mass, theta_bins, dndz, hmobject, l_beam=None):
	hmobject.set_powspec(log_meff=mass)
	return hmobject.get_kappa_prof(dndz=dndz, theta_bins=theta_bins, l_beam=l_beam)


def biased_xcorr(foo, bias, ells, dndz, hmobject):
	hmobject.set_powspec(bias1=bias)
	return hmobject.get_binned_c_ell_kg(dndz=dndz, ls=ells)


def biased_stack(foo, bias, theta_bins, dndz, hmobject, l_beam=None):
	hmobject.set_powspec(bias1=bias)
	return hmobject.get_kappa_prof(dndz=dndz, theta_bins=theta_bins, l_beam=l_beam)


# fit a cross correlation between overdensity and lensing convergence kappa
# dndz is a tuple (center zs, normalized dndz)
# xcorr is a tuple (cross power, cross power error)
def fit_xcorr(dndz, xcorr, model='mass'):
	# initialize halo model
	hmobj = hm_calcs.halomodel(zs=dndz[0])
	ells = np.arange(30, 3000)

	# fit for a constant effective mass from which bias b(M,z) is calculated
	if model == 'mass':
		partialfun = partial(mass_biased_xcorr, ells=ells, dndz=dndz, hmobject=hmobj)
		popt, pcov = curve_fit(partialfun, np.ones(len(xcorr[0])), xcorr[0], sigma=xcorr[1], absolute_sigma=True,
							   bounds=[11, 14], p0=12.5)
		hmobj.set_powspec(log_meff=popt[0])
		bestmodel = (ells, hmobj.get_c_ell_kg(dndz, ells))

	# fit for a constant bias across redshift
	elif model == 'bias':
		partialfun = partial(biased_xcorr, ells=ells, dndz=dndz, hmobject=hmobj)
		popt, pcov = curve_fit(partialfun, np.ones(len(xcorr[0])), xcorr[0], sigma=xcorr[1], absolute_sigma=True,
							   bounds=[0.5, 10], p0=2)
		hmobj.set_powspec(bias1=popt[0])
		bestmodel = (ells, hmobj.get_c_ell_kg(dndz, ells))

	return popt[0], np.sqrt(pcov)[0][0], bestmodel


# fit a lensing convergence profile
# dndz is a tuple (center zs, normalized dndz)
# stack is a tuple (theta_bins, kappa profile, profile error)
def fit_stack(dndz, stack, model='mass', l_beam=None):
	hmobj = hm_calcs.halomodel(zs=dndz[0])
	theta_bins = stack[0]
	# if no theta bins given, assume fit for peak convergence and get value of model near theta = 0
	if theta_bins is None:
		theta_bins = np.array([0.01, 0.05])

	# fit for a constant effective mass from which bias b(M,z) is calculated
	if model == 'mass':
		partialfun = partial(mass_biased_stack, theta_bins=theta_bins, dndz=dndz, hmobject=hmobj, l_beam=l_beam)
		popt, pcov = curve_fit(partialfun, np.ones(len(theta_bins)-1), stack[1], sigma=stack[2], absolute_sigma=True,
							   bounds=[11, 14], p0=12.5)
		hmobj.set_powspec(log_meff=popt[0])
		bestmodel = (theta_bins, hmobj.get_kappa_prof(dndz=dndz, theta_bins=theta_bins, l_beam=l_beam))

	# fit for a constant bias across redshift
	elif model == 'bias':
		partialfun = partial(biased_stack, theta_bins=theta_bins, dndz=dndz, hmobject=hmobj, l_beam=l_beam)
		popt, pcov = curve_fit(partialfun, np.ones(len(stack[0])), stack[1], sigma=stack[2], absolute_sigma=True,
							   bounds=[0.5, 10], p0=2)
		hmobj.set_powspec(bias1=popt[0])
		bestmodel = (theta_bins, hmobj.get_kappa_prof(dndz=dndz, theta_bins=theta_bins, l_beam=l_beam))
	return popt[0], np.sqrt(pcov)[0][0], bestmodel