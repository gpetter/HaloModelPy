import emcee
import numpy as np
from . import hm_calcs


paramdict = {'M': 0, 'sigM': 1, 'M0': 2, 'M1':3, 'alpha': 4}



def parse_params(theta, freeparam_ids):
	default_hod = np.array([12.5, 1e-3, 12.5, 13.5, 1.])
	hodparams = np.copy(default_hod)
	for j in range(len(freeparam_ids)):
		hodparams[paramdict[freeparam_ids[j]]] = theta[j]
	# if M0 not a free parameter, set equal to M_min
	if 'M0' not in freeparam_ids:
		hodparams[paramdict['M0']] = theta[0]
	# if M1 not a free parameter, set to 10x M_min
	if 'M1' not in freeparam_ids:
		hodparams[paramdict['M1']] = 1. + theta[0]
	return hodparams




# log prior function
def ln_prior(hodparams):
	mmin = hodparams[0]
	if (mmin < 11) | (mmin > 14):
		return -np.inf
	m1 = hodparams[paramdict['M1']]
	if (m1 < 12) | (m1 > 15) | (m1 < mmin):
		return -np.inf
	alpha = hodparams[paramdict['alpha']]
	if (alpha < 0.4) | (alpha > 2.5):
		return -np.inf
	return 0.



# log likelihood function
def ln_likelihood(residual, yerr):

	# if yerr is a covariance matrix, likelihood function is r.T * inv(Cov) * r
	err_shape = np.shape(yerr)
	try:
		# trying to access the second axis will throw an error if yerr is 1D, catch this error and use 1D least squares
		foo = err_shape[1]
		return -0.5 * np.dot(residual.T, np.dot(np.linalg.inv(yerr), residual))[0][0]

	# if yerr is 1D, above will throw error, use least squares
	except:
		return -0.5 * np.sum((residual / yerr) ** 2)


# log probability is prior plus likelihood
def ln_prob_cf(theta, cf, freeparam_ids, hmobj):
	anglebins = cf['theta_bins']
	y = cf['w_theta']
	yerr = cf['w_err']
	hodparams = parse_params(theta, freeparam_ids)
	prior = ln_prior(hodparams)
	if prior > -np.inf:
		hmobj.set_powspec(hodparams=hodparams)
		# keep track of derived parameters like satellite fraction, effective bias, effective mass
		#derived = (hod_model.derived_parameters(zs, dndz, theta, modeltype))
		#derived = hmobj.hm.derived_parameters(dndz=dndz)


		# get model prediciton for given parameter set
		#modelprediction = clusteringModel.angular_corr_func_in_bins(anglebins, zs=zs, dn_dz_1=dndz,
		#                                                            hodparams=theta,
		#                                                            hodmodel=modeltype)
		modelprediction = hmobj.get_binned_ang_cf(theta_bins=anglebins)

		# residual is data - model
		residual = y - modelprediction

		likely = ln_likelihood(residual, yerr)
		prob = prior + likely

		# return log_prob, along with derived parameters for this parameter set
		#return (prob,) + derived
		return prob
	else:
		return prior


# log probability is prior plus likelihood
def ln_prob_lens(theta, xcorr, freeparam_ids, hmobj):
	ell_bins = xcorr['ell_bins']
	y = xcorr['cl']
	yerr = xcorr['cl_err']
	hodparams = parse_params(theta, freeparam_ids)
	prior = ln_prior(hodparams)
	if prior > -np.inf:
		hmobj.set_powspec(hodparams=hodparams)
		# keep track of derived parameters like satellite fraction, effective bias, effective mass
		#derived = (hod_model.derived_parameters(zs, dndz, theta, modeltype))
		#derived = hmobj.hm.derived_parameters(dndz=dndz)


		# get model prediciton for given parameter set
		modelprediction = hmobj.get_binned_c_ell_kg(ell_bins)

		# residual is data - model
		residual = y - modelprediction

		likely = ln_likelihood(residual, yerr)
		prob = prior + likely

		# return log_prob, along with derived parameters for this parameter set
		#return (prob,) + derived
		return prob
	else:
		return prior



def sample_cf_space(nwalkers, niter, cf, dndz, freeparam_ids, initial_params=None, pool=None):

	ndim = len(freeparam_ids)
	#blobs_dtype = [("f_sat", float), ("b_eff", float), ("m_eff", float)]
	#if ndim == 1:
	#	blobs_dtype = blobs_dtype[1:]

	halomod_obj = hm_calcs.halomodel(dndz)


	sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob_cf,
									args=[cf, freeparam_ids, halomod_obj],
	                                pool=pool)


	# start walkers near least squares fit position with random gaussian offsets
	pos = np.array(initial_params) + 2e-1 * np.random.normal(size=(sampler.nwalkers, sampler.ndim))

	sampler.run_mcmc(pos, niter, progress=True)


	flatchain = np.array(sampler.get_chain(flat=True))
	#blobs = sampler.get_blobs(discard=10, flat=True)


	"""if ndim > 1:
		flatchain = np.hstack((
			flatchain,
			np.atleast_2d(blobs['f_sat']).T,
			np.atleast_2d(blobs['b_eff']).T,
			np.atleast_2d(blobs['m_eff']).T
		))
	else:
		flatchain = np.hstack((
			flatchain,
			np.atleast_2d(blobs['b_eff']).T,
			np.atleast_2d(blobs['m_eff']).T
		))"""



	centervals, lowerrs, higherrs = [], [], []
	#for i in range(ndim + len(blobs_dtype)):
	"""for i in range(ndim):
		post = np.percentile(flatchain[:, i], [16, 50, 84])
		q = np.diff(post)
		centervals.append(post[1])
		lowerrs.append(q[0])
		higherrs.append(q[1])"""


	#plotting.hod_corner('clustering', flatchain, ndim, binnum, nbins)

	#if binnum == nbins:
	#	flatchains = [np.load('results/chains/%s.npy' % (j+1), allow_pickle=True) for j in range(nbins)]
	#	plotting.overlapping_corners('clustering', flatchains, ndim, nbins)

	return flatchain


def sample_lens_space(nwalkers, niter, xcorr, dndz, freeparam_ids, initial_params=None, pool=None):
	ndim = len(freeparam_ids)
	#blobs_dtype = [("f_sat", float), ("b_eff", float), ("m_eff", float)]
	#if ndim == 1:
	#	blobs_dtype = blobs_dtype[1:]

	halomod_obj = hm_calcs.halomodel(dndz)


	sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob_lens, args=[xcorr, freeparam_ids, halomod_obj], pool=pool)


	# start walkers near least squares fit position with random gaussian offsets
	pos = np.array(initial_params) + 2e-1 * np.random.normal(size=(sampler.nwalkers, sampler.ndim))

	sampler.run_mcmc(pos, niter, progress=True)


	flatchain = sampler.get_chain(flat=True)
	#blobs = sampler.get_blobs(discard=10, flat=True)


	"""if ndim > 1:
		flatchain = np.hstack((
			flatchain,
			np.atleast_2d(blobs['f_sat']).T,
			np.atleast_2d(blobs['b_eff']).T,
			np.atleast_2d(blobs['m_eff']).T
		))
	else:
		flatchain = np.hstack((
			flatchain,
			np.atleast_2d(blobs['b_eff']).T,
			np.atleast_2d(blobs['m_eff']).T
		))"""

	#np.array(flatchain).dump('results/chains/%s.npy' % binnum)

	"""centervals, lowerrs, higherrs = [], [], []
	for i in range(ndim + len(blobs_dtype)):
		post = np.percentile(flatchain[:, i], [16, 50, 84])
		q = np.diff(post)
		centervals.append(post[1])
		lowerrs.append(q[0])
		higherrs.append(q[1])"""


	#plotting.hod_corner('lensing', flatchain, ndim, binnum, nbins)

	#if binnum == nbins:
		#flatchains = [np.load('results/chains/%s.npy' % (j+1), allow_pickle=True) for j in range(nbins)]
		#plotting.overlapping_corners('lensing', flatchains, ndim, nbins)

	return flatchain
