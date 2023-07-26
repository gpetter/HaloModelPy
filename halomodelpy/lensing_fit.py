import numpy as np
from . import hm_calcs
from functools import partial
from scipy.optimize import curve_fit
from . import interpolate_helper


# functions to call for fits
# take a halo mass or bias factor, return model cross correlation or kappa stack
def mass_biased_xcorr(foo, mass, hmobject, ell_bins):
	hmobject.set_powspec(log_meff=mass)
	return hmobject.get_binned_c_ell_kg(ell_bins=ell_bins)


def minmass_biased_xcorr(foo, minmass, hmobject, ell_bins):
	hmobject.set_powspec(log_m_min1=minmass)
	return hmobject.get_binned_c_ell_kg(ell_bins=ell_bins)


def mass_biased_stack(foo, mass, theta_bins, hmobject, l_beam=None):
	hmobject.set_powspec(log_meff=mass)
	return hmobject.get_binned_kappa_prof(theta_bins=theta_bins, l_beam=l_beam)


def biased_xcorr(foo, bias, hmobject, ell_bins):
	hmobject.set_powspec(bias1=bias)
	return hmobject.get_binned_c_ell_kg(ell_bins=ell_bins)


def biased_stack(foo, bias, theta_bins, hmobject, l_beam=None):
	hmobject.set_powspec(bias1=bias)
	return hmobject.get_binned_kappa_prof(theta_bins=theta_bins, l_beam=l_beam)


# fit a cross correlation between overdensity and lensing convergence kappa
# dndz is a tuple (center zs, normalized dndz)
# xcorr is a tuple (cross power, cross power error)
def fit_xcorr(dndz, xcorr, model='mass'):
	# initialize halo model
	hmobj = hm_calcs.halomodel(dndz)
	ells = np.arange(30, 3000)
	scalebins, scales, corr, err = xcorr['ell_bins'], xcorr['ell'], xcorr['cl'], xcorr['cl_err']
	unbiasedmod = hmobj.get_c_ell_kg(ells=ells)

	# fit for a constant effective mass from which bias b(M,z) is calculated
	if model == 'mass':
		partialfun = partial(mass_biased_xcorr, hmobject=hmobj, ell_bins=scalebins)
		popt, pcov = curve_fit(partialfun, np.ones(len(corr)), corr, sigma=err, absolute_sigma=True,
							   bounds=[11, 14], p0=12.5)
		hmobj.set_powspec(log_meff=popt[0])
		bestmodel = (ells, hmobj.get_c_ell_kg(ells), unbiasedmod)

	# fit for a constant bias across redshift
	elif model == 'bias':
		partialfun = partial(biased_xcorr, hmobject=hmobj, ell_bins=scalebins)
		popt, pcov = curve_fit(partialfun, np.ones(len(corr)), corr, sigma=err, absolute_sigma=True,
							   bounds=[0.5, 10], p0=2)
		hmobj.set_powspec(bias1=popt[0])
		bestmodel = (ells, hmobj.get_c_ell_kg(ells), unbiasedmod)

	elif model == 'minmass':
		partialfun = partial(minmass_biased_xcorr, hmobject=hmobj, ell_bins=scalebins)
		popt, pcov = curve_fit(partialfun, np.ones(len(corr)), corr, sigma=err, absolute_sigma=True,
							   bounds=[11, 14], p0=12.)
		hmobj.set_powspec(bias1=popt[0])
		bestmodel = (ells, hmobj.get_c_ell_kg(ells), unbiasedmod)


	return popt[0], np.sqrt(pcov)[0][0], bestmodel


def xcorr_fit_pipeline(dndz, xcorr):
	import matplotlib.pyplot as plt
	scalebins, scales, corr, err = xcorr['ell_bins'], xcorr['ell'], xcorr['cl'], xcorr['cl_err']

	fig, ax = plt.subplots(figsize=(8, 7))
	ax.scatter(scales, corr, c='k')
	ax.errorbar(scales, corr, yerr=err, ecolor='k', fmt='none')

	b, berr, b_model = fit_xcorr(dndz=dndz, xcorr=xcorr, model='bias')
	m, merr, m_model = fit_xcorr(dndz=dndz, xcorr=xcorr, model='mass')
	mmin, mmin_err, mmin_model = fit_xcorr(dndz=dndz, xcorr=xcorr, model='minmass')

	outdict = {}
	outdict['b'], outdict['sigb'] = b, berr
	outdict['M'], outdict['sigM'] = m, merr
	outdict['Mmin'], outdict['sigMmin'] = mmin, mmin_err

	ax.text(0.1, 0.2, '$b = %s \pm %s$' % (round(b, 2), round(berr, 2)), transform=plt.gca().transAxes, fontsize=15)
	ax.text(0.1, 0.15, '$log_{10}(M_{\mathrm{eff}}) = %s \pm %s$' % (round(m, 2), round(merr, 2)),
			transform=plt.gca().transAxes, fontsize=15)
	ax.text(0.1, 0.1, '$log_{10}(M_{\mathrm{min}}) = %s \pm %s$' % (round(mmin, 2), round(mmin_err, 2)),
			transform=plt.gca().transAxes, fontsize=15)

	plt.plot(b_model[0], b_model[1], c='k', ls='dotted')
	plt.plot(b_model[0], b_model[2], c='k', ls='dashed')
	plt.xscale('log')
	plt.yscale('log')

	plt.xlabel(r'$\ell$', fontsize=20)
	plt.ylabel(r'$C_{\ell}$', fontsize=20)
	plt.close()
	outdict['plot'] = fig

	return outdict


# fit a lensing convergence profile
# dndz is a tuple (center zs, normalized dndz)
# stack is a tuple (theta_bins, kappa profile, profile error)
def fit_stack(dndz, stack, model='mass', l_beam=None):
	hmobj = hm_calcs.halomodel(dndz)
	theta_bins = stack[0]
	# if no theta bins given, assume fit for peak convergence and get value of model near theta = 0
	if theta_bins is None:
		theta_bins = np.array([0.01, 0.05])

	# fit for a constant effective mass from which bias b(M,z) is calculated
	if model == 'mass':
		partialfun = partial(mass_biased_stack, theta_bins=theta_bins, hmobject=hmobj, l_beam=l_beam)
		popt, pcov = curve_fit(partialfun, np.ones(len(theta_bins)-1), stack[1], sigma=stack[2], absolute_sigma=True,
							   bounds=[11, 14], p0=12.5)
		hmobj.set_powspec(log_meff=popt[0])
		bestmodel = (theta_bins, hmobj.get_binned_kappa_prof(theta_bins=theta_bins, l_beam=l_beam))

	# fit for a constant bias across redshift
	elif model == 'bias':
		partialfun = partial(biased_stack, theta_bins=theta_bins, hmobject=hmobj, l_beam=l_beam)
		popt, pcov = curve_fit(partialfun, np.ones(len(stack[0])), stack[1], sigma=stack[2], absolute_sigma=True,
							   bounds=[0.5, 10], p0=2)
		hmobj.set_powspec(bias1=popt[0])
		bestmodel = (theta_bins, hmobj.get_binned_kappa_prof(theta_bins=theta_bins, l_beam=l_beam))
	return popt[0], np.sqrt(pcov)[0][0], bestmodel

def fitmcmc(nwalkers, niter, dndz, xcorr, freeparam_ids, initial_params):
	from . import mcmc
	from . import plotscripts
	outdict = {}
	chain = mcmc.sample_lens_space(nwalkers=nwalkers, niter=niter, dndz=dndz, xcorr=xcorr,
								 freeparam_ids=freeparam_ids, initial_params=initial_params)
	outdict['chain'] = chain
	outdict['corner'] = plotscripts.hod_corner(chain=chain, param_ids=freeparam_ids)
	#outdict['hods'] = plotscripts.hod_realizations(chain=chain, param_ids=freeparam_ids)

	return outdict