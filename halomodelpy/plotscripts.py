import matplotlib.pyplot as plt
from plottools import aesthetic


def autoclustering_fit(cf, outdict):
	if 'rp' in cf:
		angular = False
		scalebins, effscales, corr, err = cf['rp_bins'], cf['rp'], cf['wp'], cf['wp_err']
	else:
		angular = True
		scalebins, effscales, corr, err = cf['theta_bins'], cf['theta'], cf['w_theta'], cf['w_err']

	fig, ax = plt.subplots(figsize=(8, 7))
	ax.scatter(effscales, corr, c='k')
	ax.errorbar(effscales, corr, yerr=err, ecolor='k', fmt='none')
	ax.text(0.1, 0.2, '$b = %s \pm %s$' % (round(outdict['b'], 2), round(outdict['sigb'], 2)),
			transform=plt.gca().transAxes, fontsize=15)
	ax.text(0.1, 0.15, '$log_{10}(M_{\mathrm{eff}}) = %s \pm %s$' % (round(outdict['M'], 2), round(outdict['sigM'], 2)),
			transform=plt.gca().transAxes, fontsize=15)
	ax.text(0.1, 0.1,
			'$log_{10}(M_{\mathrm{min}}) = %s \pm %s$' % (round(outdict['Mmin'], 2), round(outdict['sigMmin'], 2)),
			transform=plt.gca().transAxes, fontsize=15)

	plt.plot(outdict['modscales'], outdict['dmcf'], c='k', ls='dotted')
	plt.plot(outdict['modscales'], outdict['autofitcf'], c='k', ls='dashed')
	plt.xscale('log')
	plt.yscale('log')
	if angular:
		plt.xlabel(r'$\theta$ [deg]', fontsize=20)
		plt.ylabel(r'$w(\theta)$', fontsize=20)
	else:
		plt.xlabel(r'$r_p \ [\mathrm{Mpc}/h]$', fontsize=20)
		plt.ylabel(r'$w_{p}(r_{p})$', fontsize=20)

	plt.close()
	return fig


def crossclustering_fit(cf_x, autocf, outdict):
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

	ax.text(0.1, 0.2, '$b = %s \pm %s$' % (round(outdict['b'], 2), round(outdict['sigb'], 2)),
			transform=ax.transAxes, fontsize=15)
	ax.text(0.1, 0.15, '$log_{10}(M_{\mathrm{eff}}) = %s \pm %s$' % (round(outdict['M'], 2), round(outdict['sigM'], 2)),
			transform=ax.transAxes, fontsize=15)
	ax.text(0.1, 0.1,
			'$log_{10}(M_{\mathrm{min}}) = %s \pm %s$' % (round(outdict['Mmin'], 2), round(outdict['sigMmin'], 2)),
			transform=ax.transAxes, fontsize=15)

	ax.plot(outdict['modscales'], outdict['dmcf'], c='k', ls='dotted')
	ax.plot(outdict['modscales'], outdict['autofitcf'], c='k', ls='dashed')
	ax.set_xscale('log')
	ax.set_yscale('log')

	ax2.plot(outdict['modscales'], outdict['dmcf'], c='k', ls='dotted')
	ax2.plot(outdict['modscales'], outdict['xfitcf'], c='k', ls='dashed')
	ax2.set_xscale('log')
	ax2.set_yscale('log')

	ax2.text(0.1, 0.2, '$b = %s \pm %s$' % (round(outdict['bx'], 2), round(outdict['sigbx'], 2)),
			 transform=ax2.transAxes, fontsize=15)
	ax2.text(0.1, 0.15, '$log_{10}(M_{\mathrm{eff}}) = %s \pm %s$' % (round(outdict['Mx'], 2),
																	  round(outdict['sigMx'], 2)),
			 transform=ax2.transAxes, fontsize=15)
	ax2.text(0.1, 0.1, '$log_{10}(M_{\mathrm{min}}) = %s \pm %s$' % (round(outdict['Mxmin'], 2),
																	 round(outdict['sigMxmin'], 2)),
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
	return fig