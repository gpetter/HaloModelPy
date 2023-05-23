import numpy as np
from . import hm_calcs
from . import mcmc
from . import interpolate_helper
from functools import partial
from scipy.optimize import curve_fit


# a model with a linear power spectrum with a bias corresponding to a constant halo mass
def mass_biased_cf(foo, mass, scales, hmobject, angular):
	hmobject.set_powspec(log_meff=mass)
	if angular:
		return hmobject.get_binned_ang_cf(scales)
	else:
		return hmobject.get_binned_spatial_cf(scales)


def minmass_biased_cf(foo, minmass, scales, hmobject, angular):
	hmobject.set_powspec(log_m_min1=minmass)
	if angular:
		return hmobject.get_binned_ang_cf(scales)
	else:
		return hmobject.get_binned_spatial_cf(scales)
	

# a model with a linear power spectrum with a bias of b
def biased_cf(foo, bias, scales, hmobject, angular):
	hmobject.set_powspec(bias1=bias)
	if angular:
		return hmobject.get_binned_ang_cf(scales)
	else:
		return hmobject.get_binned_spatial_cf(scales)


# same as above for cross correlations
def mass_biased_xcf(foo, mass1, mass2, scales, hmobject, angular):
	hmobject.set_powspec(log_meff=mass1, log_meff_2=mass2)
	if angular:
		return hmobject.get_binned_ang_cf(scales)
	else:
		return hmobject.get_binned_spatial_cf(scales)


def minmass_biased_xcf(foo, minmass1, minmass2, scales, hmobject, angular):
	hmobject.set_powspec(log_m_min1=minmass1, log_m_min2=minmass2)
	if angular:
		return hmobject.get_binned_ang_cf(scales)
	else:
		return hmobject.get_binned_spatial_cf(scales)


# a model with a linear power spectrum with a bias of b
def biased_xcf(foo, bias1, bias2, scales, hmobject, angular):
	hmobject.set_powspec(bias1=bias1, bias2=bias2)
	if angular:
		return hmobject.get_binned_ang_cf(scales)
	else:
		return hmobject.get_binned_spatial_cf(scales)




# fit either a projected or angular correlation function for an effective bias or halo mass
def fit_cf(dndz, cf, model='mass'):
	# initialize halo model
	hmobj = hm_calcs.halomodel(dndz)

	if 'rp' in cf:
		angular = False
		scalebins, corr, err = cf['rp_bins'], cf['wp'], cf['wp_err']
	else:
		angular = True
		scalebins, corr, err = cf['theta_bins'], cf['w_theta'], cf['w_err']
	if angular:
		modscales = np.logspace(-3, 0., 200)
		unbiasedmod = hmobj.get_ang_cf(modscales)
	else:
		modscales = np.logspace(-1, 2.3, 200)
		unbiasedmod = hmobj.get_spatial_cf(modscales)
	
	# if fitting for an effective halo mass
	if model == 'mass':
		partialfun = partial(mass_biased_cf, scales=scalebins, hmobject=hmobj, angular=angular)
		popt, pcov = curve_fit(partialfun, None, corr, sigma=err, absolute_sigma=True,
							bounds=[11, 14], p0=12.5)
		hmobj.set_powspec(log_meff=popt[0])

	# if fitting for an effective bias
	elif model == 'bias':
		partialfun = partial(biased_cf, scales=scalebins, hmobject=hmobj, angular=angular)
		popt, pcov = curve_fit(partialfun, None, corr, sigma=err, absolute_sigma=True,
							bounds=[0.5, 30], p0=2)
		hmobj.set_powspec(bias1=popt[0])
	elif model == 'minmass':
		partialfun = partial(minmass_biased_cf, scales=scalebins, hmobject=hmobj, angular=angular)
		popt, pcov = curve_fit(partialfun, None, corr, sigma=err, absolute_sigma=True,
							   bounds=[11, 14], p0=12.)
		hmobj.set_powspec(log_m_min1=popt[0])



	# or do full hod modeling with mcmc, not implemented
	else:
		centervals, lowerrs, higherrs = mcmc.sample_cf_space
	if angular:
		# return the best fit model on a grid for plotting purposes
		bestmodel = (modscales, hmobj.get_ang_cf(modscales), unbiasedmod)
	else:
		bestmodel = (modscales, hmobj.get_spatial_cf(radii=modscales), unbiasedmod)

	return popt[0], np.sqrt(pcov)[0][0], bestmodel


# fit either a projected or angular correlation function for an effective bias or halo mass
def fit_xcf(dndz_x, cf_x, dndz_auto, autocf, model='mass'):
	# initialize halo model
	hmobj = hm_calcs.halomodel(dndz1=dndz_auto, dndz2=dndz_x)

	if 'rp' in cf_x:
		angular = False
		scalebins, xcorr, xerr = cf_x['rp_bins'], cf_x['wp'], cf_x['wp_err']

	else:
		angular = True
		scalebins, xcorr, xerr = cf_x['theta_bins'], cf_x['w_theta'], cf_x['w_err']

	if angular:
		modscales = np.logspace(-3, 0., 200)
		unbiasedmod = hmobj.get_ang_cf(modscales)
	else:
		modscales = np.logspace(-1, 2.3, 200)
		unbiasedmod = hmobj.get_spatial_cf(radii=modscales)

	autofit = fit_cf(dndz=dndz_auto, cf=autocf, model=model)
	par_auto, err_auto = autofit[0], autofit[1]

	# if fitting for an effective halo mass
	if model == 'mass':
		partialfun = partial(mass_biased_xcf, mass2=par_auto, dndz1=dndz_x, dndz2=dndz_auto, scales=scalebins,
							 hmobject=hmobj, angular=angular)
		popt, pcov = curve_fit(partialfun, None, xcorr, sigma=xerr, absolute_sigma=True,
							   bounds=[11, 14], p0=12.5)
		hmobj.set_powspec(log_meff=popt[0])

	# if fitting for an effective bias
	elif model == 'bias':
		partialfun = partial(biased_xcf, bias2=par_auto, dndz1=dndz_x, dndz2=dndz_auto, scales=scalebins,
							 hmobject=hmobj, angular=angular)
		popt, pcov = curve_fit(partialfun, None, xcorr, sigma=xerr, absolute_sigma=True,
							   bounds=[0.5, 30], p0=2)
		hmobj.set_powspec(bias1=popt[0])
	elif model == 'minmass':
		partialfun = partial(minmass_biased_xcf, minmass2=par_auto, dndz1=dndz_x, dndz2=dndz_auto, scales=scalebins,
							 hmobject=hmobj, angular=angular)
		popt, pcov = curve_fit(partialfun, None, xcorr, sigma=xerr, absolute_sigma=True,
							   bounds=[11, 14], p0=12.)
		hmobj.set_powspec(log_m_min1=popt[0])

	# or do full hod modeling with mcmc, not implemented
	else:
		centervals, lowerrs, higherrs = mcmc.sample_cf_space

	if angular:
		# return the best fit model on a grid for plotting purposes
		bestmodel = (modscales, hmobj.get_ang_cf(modscales), unbiasedmod)
	else:
		bestmodel = (modscales, hmobj.get_spatial_cf(radii=modscales), unbiasedmod)

	# Figure out how to incorporate errors on reference population
	return popt[0], np.sqrt(pcov)[0][0], bestmodel


# fit for bias, effective mass, minimum mass, and return diagnostic plot
def fit_pipeline(dndz, cf, dndL=None):
	import matplotlib.pyplot as plt

	if 'rp' in cf:
		angular = False
		scalebins, effscales, corr, err = cf['rp_bins'], cf['rp'], cf['wp'], cf['wp_err']
	else:
		angular = True
		scalebins, effscales, corr, err = cf['theta_bins'], cf['theta'], cf['w_theta'], cf['w_err']

	fig, ax = plt.subplots(figsize=(8, 7))
	ax.scatter(effscales, corr, c='k')
	ax.errorbar(effscales, corr, yerr=err, ecolor='k', fmt='none')

	outdict = {}
	b, berr, b_model = fit_cf(dndz=dndz, cf=cf, model='bias')
	m, merr, m_model = fit_cf(dndz=dndz, cf=cf, model='mass')
	mmin, mmin_err, mmin_model = fit_cf(dndz=dndz, cf=cf, model='minmass')

	outdict['b'], outdict['sigb'] = b, berr
	outdict['M'], outdict['sigM'] = m, merr
	outdict['Mmin'], outdict['sigMmin'] = mmin, mmin_err
	if dndL is not None:
		from . import luminosityfunction
		fduty = luminosityfunction.occupation_fraction(dndL, dndz, logminmasses=mmin)[0]
		ax.text(0.1, 0.05, r'$f_{\mathrm{duty}} = %s$' % (round(fduty, 4)),
				transform=plt.gca().transAxes, fontsize=15)
	
	ax.text(0.1, 0.2, '$b = %s \pm %s$' % (round(b, 2), round(berr, 2)),
			transform=plt.gca().transAxes, fontsize=15)
	ax.text(0.1, 0.15, '$log_{10}(M_{\mathrm{eff}}) = %s \pm %s$' % (round(m, 2), round(merr, 2)),
			transform=plt.gca().transAxes, fontsize=15)
	ax.text(0.1, 0.1, '$log_{10}(M_{\mathrm{min}}) = %s \pm %s$' % (round(mmin, 2), round(mmin_err, 2)),
			transform=plt.gca().transAxes, fontsize=15)
	
	plt.plot(b_model[0], b_model[1], c='k', ls='dotted')
	plt.plot(b_model[0], b_model[2], c='k', ls='dashed')
	plt.xscale('log')
	plt.yscale('log')
	if angular:
		plt.xlabel(r'$\theta$ [deg]', fontsize=20)
		plt.ylabel(r'$w(\theta)$', fontsize=20)
	else:
		plt.xlabel(r'$r_p \ [\mathrm{Mpc}/h]$', fontsize=20)
		plt.ylabel(r'$w_{p}(r_{p})$', fontsize=20)
	plt.close()
	outdict['plot'] = fig
	
	return outdict


# fit for bias, effective mass, minimum mass, and return diagnostic plot
def xfit_pipeline(dndz_x, cf_x, dndz_auto, autocf):
	import matplotlib.pyplot as plt
	from plottools import aesthetic

	if 'rp' in cf_x:
		angular = False
		effscales, xcorr, xerr = cf_x['rp'], cf_x['wp'], cf_x['wp_err']
		autocorr, autoerr = autocf['wp'], autocf['wp_err']

	else:
		angular = True
		effscales, xcorr, xerr = cf_x['theta'], cf_x['w_theta'], cf_x['w_err']
		autocorr, autoerr = autocf['w_theta'], autocf['w_err']

	fig, (ax, ax2) = plt.subplots(figsize=(16, 7), ncols=2, sharey=True)
	ax.scatter(effscales, autocorr, c='k')
	ax.errorbar(effscales, autocorr, yerr=autoerr, ecolor='k', fmt='none')
	ax2.scatter(effscales, xcorr, c='k')
	ax2.errorbar(effscales, xcorr, yerr=xerr, ecolor='k', fmt='none')

	outdict = {}
	b, berr, b_model = fit_cf(dndz=dndz_auto, cf=autocf, model='bias')
	m, merr, m_model = fit_cf(dndz=dndz_auto, cf=autocf, model='mass')
	mmin, mmin_err, mmin_model = fit_cf(dndz=dndz_auto, cf=autocf, model='minmass')

	outdict['b'], outdict['sigb'] = b, berr
	outdict['M'], outdict['sigM'] = m, merr
	outdict['Mmin'], outdict['sigMmin'] = mmin, mmin_err

	ax.text(0.1, 0.2, '$b = %s \pm %s$' % (round(b, 2), round(berr, 2)),
			transform=ax.transAxes, fontsize=15)
	ax.text(0.1, 0.15, '$log_{10}(M_{\mathrm{eff}}) = %s \pm %s$' % (round(m, 2), round(merr, 2)),
			transform=ax.transAxes, fontsize=15)
	ax.text(0.1, 0.1, '$log_{10}(M_{\mathrm{min}}) = %s \pm %s$' % (round(mmin, 2), round(mmin_err, 2)),
			transform=ax.transAxes, fontsize=15)

	ax.plot(b_model[0], b_model[1], c='k', ls='dotted')
	ax.plot(b_model[0], b_model[2], c='k', ls='dashed')
	ax.set_xscale('log')
	ax.set_yscale('log')

	bx, bxerr, bx_model = fit_xcf(dndz_x=dndz_x, cf_x=cf_x, dndz_auto=dndz_auto, autocf=autocf, model='bias')
	mx, mxerr, mx_model = fit_xcf(dndz_x=dndz_x, cf_x=cf_x, dndz_auto=dndz_auto, autocf=autocf, model='mass')
	mxmin, mxmin_err, mxmin_model = fit_xcf(dndz_x=dndz_x, cf_x=cf_x, dndz_auto=dndz_auto,
											autocf=autocf, model='minmass')

	outdict['bx'], outdict['sigbx'] = bx, bxerr
	outdict['Mx'], outdict['sigMx'] = mx, mxerr
	outdict['Mxmin'], outdict['sigMxmin'] = mxmin, mxmin_err

	ax2.plot(bx_model[0], bx_model[1], c='k', ls='dotted')
	ax2.plot(bx_model[0], bx_model[2], c='k', ls='dashed')
	ax2.set_xscale('log')
	ax2.set_yscale('log')

	ax2.text(0.1, 0.2, '$b = %s \pm %s$' % (round(bx, 2), round(bxerr, 2)),
			transform=ax2.transAxes, fontsize=15)
	ax2.text(0.1, 0.15, '$log_{10}(M_{\mathrm{eff}}) = %s \pm %s$' % (round(mx, 2), round(mxerr, 2)),
			transform=ax2.transAxes, fontsize=15)
	ax2.text(0.1, 0.1, '$log_{10}(M_{\mathrm{min}}) = %s \pm %s$' % (round(mxmin, 2), round(mxmin_err, 2)),
			transform=ax2.transAxes, fontsize=15)


	if angular:
		ax.set_xlabel(r'$\theta$ [deg]', fontsize=20)
		ax2.set_xlabel(r'$\theta$ [deg]', fontsize=20)
		ax.set_ylabel(r'$w(\theta)$', fontsize=20)
	else:
		ax.set_xlabel(r'$r_p \ [\mathrm{Mpc}/h]$', fontsize=20)
		ax2.set_xlabel(r'$r_p \ [\mathrm{Mpc}/h]$', fontsize=20)
		ax.set_ylabel(r'$w_{p}(r_{p})$', fontsize=20)
	plt.subplots_adjust(wspace=0)
	plt.close()
	outdict['plot'] = fig

	return outdict