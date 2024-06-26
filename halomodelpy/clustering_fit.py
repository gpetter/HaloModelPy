import numpy as np
from . import hm_calcs
from functools import partial
from scipy.optimize import curve_fit


# a model with a linear power spectrum with a bias corresponding to a constant halo mass
def mass_biased_cf(foo, mass, scales, hmobject, angular, idx):
	hmobject.set_powspec(log_meff=mass)
	if angular:
		return hmobject.get_binned_ang_cf(scales)[idx]
	else:
		return hmobject.get_binned_spatial_cf(scales)[idx]


def minmass_biased_cf(foo, minmass, scales, hmobject, angular, idx):
	hmobject.set_powspec(log_m_min1=minmass)
	if angular:
		return hmobject.get_binned_ang_cf(scales)[idx]
	else:
		return hmobject.get_binned_spatial_cf(scales)[idx]
	

# a model with a linear power spectrum with a bias of b
def biased_cf(foo, bias, scales, hmobject, angular, idx):
	hmobject.set_powspec(bias1=bias)
	if angular:
		return hmobject.get_binned_ang_cf(scales)[idx]
	else:
		return hmobject.get_binned_spatial_cf(scales)[idx]


# same as above for cross correlations
def mass_biased_xcf(foo, mass1, mass2, scales, hmobject, angular, idx):
	hmobject.set_powspec(log_meff=mass1, log_meff_2=mass2)
	if angular:
		return hmobject.get_binned_ang_cf(scales)[idx]
	else:
		return hmobject.get_binned_spatial_cf(scales)[idx]


def minmass_biased_xcf(foo, minmass1, minmass2, scales, hmobject, angular, idx):
	hmobject.set_powspec(log_m_min1=minmass1, log_m_min2=minmass2)
	if angular:
		return hmobject.get_binned_ang_cf(scales)[idx]
	else:
		return hmobject.get_binned_spatial_cf(scales)[idx]


# a model with a linear power spectrum with a bias of b
def biased_xcf(foo, bias1, bias2, scales, hmobject, angular, idx):
	hmobject.set_powspec(bias1=bias1, bias2=bias2)
	if angular:
		return hmobject.get_binned_ang_cf(scales)[idx]
	else:
		return hmobject.get_binned_spatial_cf(scales)[idx]


def hod_3param_cf(foo, mmin, m1, alpha, scales, hmobject, angular):
	hmobject.set_powspec(hodparams=[mmin, 0.0001, mmin, m1, alpha])
	if angular:
		return hmobject.get_binned_ang_cf(scales)
	else:
		return hmobject.get_binned_spatial_cf(scales)


def hod_2param_cf(foo, mmin, m1, scales, hmobject, angular, alpha=1.):
	return hod_3param_cf(foo, mmin, m1, alpha, scales, hmobject, angular)

# fit either a projected or angular correlation function for an effective bias or halo mass
def fit_cf(dndz, cf, model='mass'):
	# initialize halo model
	hmobj = hm_calcs.halomodel(dndz)
	outdict = {}

	if 'rp' in cf:
		angular = False
		try:
			scalebins, corr, err = cf['rp_bins'], cf['wp'], cf['wp_err']
		except:
			print('Using Poisson errors')
			scalebins, corr, err = cf['rp_bins'], cf['wp'], cf['wp_poisson_err']
	else:
		angular = True
		try:
			scalebins, corr, err = cf['theta_bins'], cf['w_theta'], cf['w_err']
		except:
			print('Using Poisson errors')
			scalebins, corr, err = cf['theta_bins'], cf['w_theta'], cf['w_err_poisson']
	if angular:
		modscales = np.logspace(-2.5, 0.25, 200)
		unbiasedmod = hmobj.get_ang_cf(modscales)
	else:
		modscales = np.logspace(-1, 2.3, 200)
		unbiasedmod = hmobj.get_spatial_cf(modscales)
	# dont use nans or zero errors in the fit
	goodidx = np.where(np.isfinite(corr) & (np.isfinite(err)) & (err > 0))[0]
	corr, err = corr[goodidx], err[goodidx]
	
	# if fitting for an effective halo mass
	if model == 'mass':
		partialfun = partial(mass_biased_cf, scales=scalebins, hmobject=hmobj, angular=angular, idx=goodidx)
		popt, pcov = curve_fit(partialfun, None, corr, sigma=err, absolute_sigma=True,
							bounds=[11, 14.5], p0=12.5)
		hmobj.set_powspec(log_meff=popt[0])
		outdict['M'], outdict['sigM'] = popt[0], np.sqrt(pcov)[0][0]

	# if fitting for an effective bias
	elif model == 'bias':
		partialfun = partial(biased_cf, scales=scalebins, hmobject=hmobj, angular=angular, idx=goodidx)
		popt, pcov = curve_fit(partialfun, None, corr, sigma=err, absolute_sigma=True,
							bounds=[0.5, 100], p0=2)
		hmobj.set_powspec(bias1=popt[0])
		outdict['b'], outdict['sigb'] = popt[0], np.sqrt(pcov)[0][0]

	elif model == 'minmass':
		partialfun = partial(minmass_biased_cf, scales=scalebins, hmobject=hmobj, angular=angular, idx=goodidx)
		popt, pcov = curve_fit(partialfun, None, corr, sigma=err, absolute_sigma=True,
							   bounds=[11, 14.5], p0=12.)
		hmobj.set_powspec(log_m_min1=popt[0])
		outdict['Mmin'], outdict['sigMmin'] = popt[0], np.sqrt(pcov)[0][0]

	else:
		print('Options are mass, bias, minmass')
	if angular:
		# return the best fit model on a grid for plotting purposes
		bestmodel = hmobj.get_ang_cf(modscales)
	else:
		bestmodel = hmobj.get_spatial_cf(radii=modscales)

	outdict['modscales'] = modscales
	outdict['dmcf'] = unbiasedmod
	outdict['autofitcf'] = bestmodel

	return outdict


# fit either a projected or angular correlation function for an effective bias or halo mass
def fit_xcf(dndz_x, cf_x, dndz_auto, autocf, model='mass'):
	# initialize halo model
	hmobj = hm_calcs.halomodel(dndz1=dndz_auto, dndz2=dndz_x)

	if 'rp' in cf_x:
		angular = False
		try:
			scalebins, xcorr, xerr = cf_x['rp_bins'], cf_x['wp'], cf_x['wp_err']
		except:
			print('Using Poisson errors')
			scalebins, xcorr, xerr = cf_x['rp_bins'], cf_x['wp'], cf_x['wp_poisson_err']

	else:
		angular = True
		try:
			scalebins, xcorr, xerr = cf_x['theta_bins'], cf_x['w_theta'], cf_x['w_err']
		except:
			print('Using Poisson errors')
			scalebins, xcorr, xerr = cf_x['theta_bins'], cf_x['w_theta'], cf_x['w_err_poisson']

	# dont use nans or zero errors in the fit
	goodidx = np.where(np.isfinite(xcorr) & (np.isfinite(xerr)) & (xerr > 0))[0]
	xcorr, xerr = xcorr[goodidx], xerr[goodidx]

	if angular:
		modscales = np.logspace(-2.5, 0.25, 200)
		unbiasedmod = hmobj.get_ang_cf(modscales)
	else:
		modscales = np.logspace(-1, 2.3, 200)
		unbiasedmod = hmobj.get_spatial_cf(radii=modscales)

	autofit = fit_cf(dndz=dndz_auto, cf=autocf, model=model)


	outdict = {}
	# if fitting for an effective halo mass
	if model == 'mass':
		par_auto, err_auto = autofit['M'], autofit['sigM']
		partialfun = partial(mass_biased_xcf, mass2=par_auto, scales=scalebins,
							 hmobject=hmobj, angular=angular, idx=goodidx)
		popt, pcov = curve_fit(partialfun, None, xcorr, sigma=xerr, absolute_sigma=True,
							   bounds=[11, 14.5], p0=12.5)
		hmobj.set_powspec(log_meff=par_auto, log_meff_2=popt[0])
		outdict['Mx'] = popt[0]
		outdict['sigMx'] = np.sqrt(pcov)[0][0]

	# if fitting for an effective bias
	elif model == 'bias':
		par_auto, err_auto = autofit['b'], autofit['sigb']
		partialfun = partial(biased_xcf, bias2=par_auto, scales=scalebins,
							 hmobject=hmobj, angular=angular, idx=goodidx)
		popt, pcov = curve_fit(partialfun, None, xcorr, sigma=xerr, absolute_sigma=True,
							   bounds=[0.5, 30], p0=2)
		hmobj.set_powspec(bias1=par_auto, bias2=popt[0])
		outdict['bx'] = popt[0]
		outdict['sigbx'] = np.sqrt(pcov)[0][0]
	elif model == 'minmass':
		par_auto, err_auto = autofit['Mmin'], autofit['sigMmin']
		partialfun = partial(minmass_biased_xcf, minmass2=par_auto, scales=scalebins,
							 hmobject=hmobj, angular=angular, idx=goodidx)
		popt, pcov = curve_fit(partialfun, None, xcorr, sigma=xerr, absolute_sigma=True,
							   bounds=[11, 14.5], p0=12.)
		hmobj.set_powspec(log_m_min1=par_auto, log_m_min2=popt[0])
		outdict['Mxmin'] = popt[0]
		outdict['sigMxmin'] = np.sqrt(pcov)[0][0]

	# or do full hod modeling with mcmc, not implemented
	else:
		print('Options are mass, bias, minmass')

	if angular:
		# return the best fit model on a grid for plotting purposes
		bestmodel = hmobj.get_ang_cf(modscales)
	else:
		bestmodel = hmobj.get_spatial_cf(radii=modscales)
	outdict['modscales'] = modscales
	outdict['dmcf'] = unbiasedmod
	outdict['xfitcf'] = bestmodel
	# Figure out how to incorporate errors on reference population
	return outdict


# fit for bias, effective mass, minimum mass, and return diagnostic plot
def fit_pipeline(dndz, cf, dndL=None):
	from . import plotscripts

	outdict = {}
	outdict.update(fit_cf(dndz=dndz, cf=cf, model='bias'))
	outdict.update(fit_cf(dndz=dndz, cf=cf, model='mass'))
	outdict.update(fit_cf(dndz=dndz, cf=cf, model='minmass'))

	if dndL is not None:
		from . import luminosityfunction
		fduty = luminosityfunction.occupation_fraction(dndL, dndz, logminmasses=outdict['Mmin'])[0]
		outdict['fduty'] = fduty
		#ax.text(0.1, 0.05, r'$f_{\mathrm{duty}} = %s$' % (round(fduty, 4)),
		#		transform=plt.gca().transAxes, fontsize=15)

	outdict['plot'] = plotscripts.autoclustering_fit(cf, outdict)
	
	return outdict


# fit for bias, effective mass, minimum mass, and return diagnostic plot
def xfit_pipeline(dndz_x, cf_x, dndz_auto, autocf):
	from . import plotscripts

	outdict = {}
	# auto fits
	outdict.update(fit_cf(dndz=dndz_auto, cf=autocf, model='bias'))
	outdict.update(fit_cf(dndz=dndz_auto, cf=autocf, model='mass'))
	outdict.update(fit_cf(dndz=dndz_auto, cf=autocf, model='minmass'))
	# cross fits
	outdict.update(fit_xcf(dndz_x=dndz_x, cf_x=cf_x, dndz_auto=dndz_auto, autocf=autocf, model='bias'))
	outdict.update(fit_xcf(dndz_x=dndz_x, cf_x=cf_x, dndz_auto=dndz_auto, autocf=autocf, model='mass'))
	outdict.update(fit_xcf(dndz_x=dndz_x, cf_x=cf_x, dndz_auto=dndz_auto, autocf=autocf, model='minmass'))

	outdict['plot'] = plotscripts.crossclustering_fit(cf_x=cf_x, autocf=autocf, outdict=outdict)

	return outdict

def fitmcmc(nwalkers, niter, dndz, cf, freeparam_ids, initial_params, prior_dict=None):
	from . import mcmc
	from . import plotscripts
	outdict = {}
	mcmc_obj = mcmc.zhengHODsampler()
	if prior_dict is not None:
		mcmc_obj.update_priors(priordict=prior_dict)
	chain = mcmc_obj.sample_cf_space(nwalkers=nwalkers, niter=niter, dndz=dndz, cf=cf,
								 freeparam_ids=freeparam_ids, initial_params=initial_params)
	outdict['chain'] = chain
	outdict['corner'] = plotscripts.hod_corner(chain=chain, param_ids=freeparam_ids)
	#outdict['hods'] = plotscripts.hod_realizations(chain=chain, param_ids=freeparam_ids)

	return outdict

def fit_xcorr_mcmc(nwalkers, niter, dndz_x, dndz_ref, xcf, ref_hod_chain, freeparam_ids,
				   initial_params, prior_dict=None):
	from . import mcmc
	outdict = {}
	mcmc_obj = mcmc.zhengHODsampler()
	if prior_dict is not None:
		mcmc_obj.update_priors(priordict=prior_dict)
	chain = mcmc_obj.sample_xcf_space(nwalkers=nwalkers, niter=niter, xcf=xcf, dndz_x=dndz_x, dndz_ref=dndz_ref,
								  ref_hod_chain=ref_hod_chain,
								  freeparam_ids=freeparam_ids, initial_params=initial_params)
	outdict['chain'] = chain
	#outdict['corner'] = plotscripts.hod_corner(chain=chain, param_ids=freeparam_ids)
	return outdict