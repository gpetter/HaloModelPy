import numpy as np
from . import hm_calcs
from . import mcmc
from . import interpolate_helper
from functools import partial
from scipy.optimize import curve_fit


# a model with a linear power spectrum with a bias corresponding to a constant halo mass
def mass_biased_cf(foo, mass, scales, dndz, hmobject, angular):
	hmobject.set_powspec(log_meff=mass)
	if angular:
		return hmobject.get_binned_ang_cf(dndz, scales)
	else:
		return hmobject.get_binned_spatial_cf(dndz, scales)

def minmass_biased_cf(foo, minmass, scales, dndz, hmobject, angular):
	hmobject.set_powspec(log_m_min1=minmass)
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
	if angular:
		modscales = np.logspace(-3, 0., 200)
		unbiasedmod = hmobj.get_ang_cf(dndz, modscales)
	else:
		modscales = np.logspace(-1, 2.3, 200)
		unbiasedmod = hmobj.get_spatial_cf(dndz, radii=modscales)
	
	# if fitting for an effective halo mass
	if model == 'mass':
		partialfun = partial(mass_biased_cf, scales=cf[0], dndz=dndz, hmobject=hmobj, angular=angular)
		popt, pcov = curve_fit(partialfun, np.ones(len(cf[1])), cf[1], sigma=cf[2], absolute_sigma=True,
							bounds=[11, 14], p0=12.5)
		hmobj.set_powspec(log_meff=popt[0])
		if angular:
			# return the best fit model on a grid for plotting purposes
			bestmodel = (modscales, hmobj.get_ang_cf(dndz, modscales), unbiasedmod)
		else:
			bestmodel = (modscales, hmobj.get_spatial_cf(dndz, radii=modscales), unbiasedmod)

	# if fitting for an effective bias
	elif model == 'bias':
		partialfun = partial(biased_cf, scales=cf[0], dndz=dndz, hmobject=hmobj, angular=angular)
		popt, pcov = curve_fit(partialfun, np.ones(len(cf[1])), cf[1], sigma=cf[2], absolute_sigma=True,
							bounds=[0.5, 30], p0=2)
		hmobj.set_powspec(bias1=popt[0])
		if angular:
			bestmodel = (modscales, hmobj.get_ang_cf(dndz, modscales), unbiasedmod)
		else:
			bestmodel = (modscales, hmobj.get_spatial_cf(dndz, radii=modscales), unbiasedmod)
	elif model == 'minmass':
		partialfun = partial(minmass_biased_cf, scales=cf[0], dndz=dndz, hmobject=hmobj, angular=angular)
		popt, pcov = curve_fit(partialfun, np.ones(len(cf[1])), cf[1], sigma=cf[2], absolute_sigma=True, 
							   bounds=[11, 14], p0=12.)
		hmobj.set_powspec(log_m_min1=popt[0])
		if angular:
			# return the best fit model on a grid for plotting purposes
			bestmodel = (modscales, hmobj.get_ang_cf(dndz, modscales), unbiasedmod)
		else:
			bestmodel = (modscales, hmobj.get_spatial_cf(dndz, radii=modscales), unbiasedmod)


	# or do full hod modeling with mcmc, not implemented
	else:
		centervals, lowerrs, higherrs = mcmc.sample_cf_space

	return popt[0], np.sqrt(pcov)[0][0], bestmodel


# fit for bias, effective mass, minimum mass, and return diagnostic plot
def fit_pipeline(dndz, cf, angular=False):
	import matplotlib.pyplot as plt
	try:
		plt.style.use(['science', '/home/graysonpetter/ssd/Dartmouth/mpl_style/pub.mplstyle'])
	except:
		print()
	bincenters = interpolate_helper.bin_centers(cf[0], 'geo_mean')
	fig, ax = plt.subplots(figsize=(8, 7))
	ax.scatter(bincenters, cf[1], c='k')
	ax.errorbar(bincenters, cf[1], yerr=cf[2], ecolor='k', fmt='none')
	
	b, berr, b_model = fit_cf(dndz=dndz, cf=cf, angular=angular, model='bias')
	m, merr, m_model = fit_cf(dndz=dndz, cf=cf, angular=angular, model='mass')
	mmin, mmin_err, mmin_model = fit_cf(dndz=dndz, cf=cf, angular=angular, model='minmass')
	
	ax.text(0.1, 0.2, '$b = %s \pm %s$' % (round(b, 2), round(berr, 2)), transform=plt.gca().transAxes, fontsize=15)
	ax.text(0.1, 0.15, '$log_{10}(M_{\mathrm{eff}}) = %s \pm %s$' % (round(m, 2), round(merr, 2)), transform=plt.gca().transAxes, fontsize=15)
	ax.text(0.1, 0.1, '$log_{10}(M_{\mathrm{min}}) = %s \pm %s$' % (round(mmin, 2), round(mmin_err, 2)), transform=plt.gca().transAxes, fontsize=15)
	
	plt.plot(b_model[0], b_model[1], c='k', ls='dotted')
	plt.plot(b_model[0], b_model[2], c='k', ls='dashed')
	plt.xscale('log')
	plt.yscale('log')
	if angular:
		plt.xlabel(r'$\theta$ [deg]', fontsize=20)
		plt.ylabel(r'$w(\theta)$', fontsize=20)
	else:
		plt.xlabel(r'$r_p [\mathrm{Mpc}/h]$', fontsize=20)
		plt.ylabel(r'$w_{p}(r_{p})$', fontsize=20)
	
	return ax