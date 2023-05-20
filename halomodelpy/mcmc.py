import emcee
import numpy as np
from . import hm_calcs




# log prior function
def ln_prior(theta):
	if len(theta) == 1:
		mmin = theta
		if (mmin > 11) & (mmin < 14):
			return 0.
		else:
			return -np.inf
	elif len(theta) == 2:
		mmin, alpha = theta
		# make flat priors on minimum mass and satellite power law
		if (mmin > 11.) & (mmin < 14.) & (alpha > 0) & (alpha < 2):
			return 0.
		else:
			return -np.inf
	elif len(theta) == 3:
		mmin, alpha, m1 = theta
		# make flat priors on minimum mass and satellite power law
		if (mmin > 11.) & (mmin < 14.) & (alpha > 0) & (alpha < 2) & (m1 > mmin) & (m1 < 15.):
			return 0.
		else:

			return -np.inf



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
def ln_prob_cf(theta, cf, dndz, modeltype, hmobj):
	anglebins, y, yerr = cf
	zs, dndz = dndz
	prior = ln_prior(theta)

	hmobj.set_powspec(hodparams=theta, modeltype=modeltype)
	# keep track of derived parameters like satellite fraction, effective bias, effective mass
	#derived = (hod_model.derived_parameters(zs, dndz, theta, modeltype))
	derived = hmobj.hm.derived_parameters(dndz=dndz)



	# get model prediciton for given parameter set
	#modelprediction = clusteringModel.angular_corr_func_in_bins(anglebins, zs=zs, dn_dz_1=dndz,
	#                                                            hodparams=theta,
	#                                                            hodmodel=modeltype)
	modelprediction = hmobj.get_binned_ang_cf(dndz=(zs, dndz), theta_bins=anglebins)

	# residual is data - model
	residual = y - modelprediction

	likely = ln_likelihood(residual, yerr)
	prob = prior + likely

	# return log_prob, along with derived parameters for this parameter set
	return (prob,) + derived


# log probability is prior plus likelihood
def ln_prob_lens(theta, ell_bins, y, yerr, zs, dndz, modeltype, hmobj):
	prior = ln_prior(theta)

	hmobj.set_powspec(hodparams=theta, modeltype=modeltype)
	# keep track of derived parameters like satellite fraction, effective bias, effective mass
	#derived = (hod_model.derived_parameters(zs, dndz, theta, modeltype))
	derived = hmobj.hm.derived_parameters(dndz=dndz)



	# get model prediciton for given parameter set
	modelprediction = hmobj.get_binned_c_ell_kg(dndz=(zs, dndz), ls=(1+np.arange(3000)))

	# residual is data - model
	residual = y - modelprediction

	likely = ln_likelihood(residual, yerr)
	prob = prior + likely

	# return log_prob, along with derived parameters for this parameter set
	return (prob,) + derived



def sample_cf_space(nwalkers, niter, cf, dndz, modeltype, initial_params=None, pool=None):

	ndim = int(modeltype[0])
	blobs_dtype = [("f_sat", float), ("b_eff", float), ("m_eff", float)]
	if ndim == 1:
		blobs_dtype = blobs_dtype[1:]

	halomod_obj = hm_calcs.halomodel(zs=dndz[0])


	sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob_cf,
									args=[cf, dndz, modeltype, halomod_obj],
									blobs_dtype=blobs_dtype,
	                                pool=pool)


	if initial_params is None:
		initial_params = [12.5, 0.5, 13.5]
		initial_params = initial_params[:ndim]

	# start walkers near least squares fit position with random gaussian offsets
	pos = np.array(initial_params) + 2e-1 * np.random.normal(size=(sampler.nwalkers, sampler.ndim))

	sampler.run_mcmc(pos, niter, progress=True)


	flatchain = sampler.get_chain(discard=10, flat=True)
	blobs = sampler.get_blobs(discard=10, flat=True)


	if ndim > 1:
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
		))

	#np.array(flatchain).dump('results/chains/%s.npy' % binnum)

	centervals, lowerrs, higherrs = [], [], []
	for i in range(ndim + len(blobs_dtype)):
		post = np.percentile(flatchain[:, i], [16, 50, 84])
		q = np.diff(post)
		centervals.append(post[1])
		lowerrs.append(q[0])
		higherrs.append(q[1])


	#plotting.hod_corner('clustering', flatchain, ndim, binnum, nbins)

	#if binnum == nbins:
	#	flatchains = [np.load('results/chains/%s.npy' % (j+1), allow_pickle=True) for j in range(nbins)]
	#	plotting.overlapping_corners('clustering', flatchains, ndim, nbins)

	return centervals, lowerrs, higherrs


def sample_lens_space(binnum, nbins, nwalkers, ndim, niter, ell_bins, y, yerr,
                 zs, dndz, modeltype, initial_params=None, pool=None):

	blobs_dtype = [("f_sat", float), ("b_eff", float), ("m_eff", float)]
	if ndim == 1:
		blobs_dtype = blobs_dtype[1:]

	halomod_obj = hm_calcs.halomodel(zs=zs)


	sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob_lens, args=[ell_bins, y, yerr, zs, dndz,
	                                                               modeltype, halomod_obj], blobs_dtype=blobs_dtype,
	                                pool=pool)


	if initial_params is None:
		initial_params = [12.5, 0.5, 13.5]
		initial_params = initial_params[:ndim]

	# start walkers near least squares fit position with random gaussian offsets
	pos = np.array(initial_params) + 2e-1 * np.random.normal(size=(sampler.nwalkers, sampler.ndim))

	sampler.run_mcmc(pos, niter, progress=True)


	flatchain = sampler.get_chain(discard=10, flat=True)
	blobs = sampler.get_blobs(discard=10, flat=True)


	if ndim > 1:
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
		))

	#np.array(flatchain).dump('results/chains/%s.npy' % binnum)

	centervals, lowerrs, higherrs = [], [], []
	for i in range(ndim + len(blobs_dtype)):
		post = np.percentile(flatchain[:, i], [16, 50, 84])
		q = np.diff(post)
		centervals.append(post[1])
		lowerrs.append(q[0])
		higherrs.append(q[1])


	#plotting.hod_corner('lensing', flatchain, ndim, binnum, nbins)

	#if binnum == nbins:
		#flatchains = [np.load('results/chains/%s.npy' % (j+1), allow_pickle=True) for j in range(nbins)]
		#plotting.overlapping_corners('lensing', flatchains, ndim, nbins)

	return centervals, lowerrs, higherrs
