import emcee
import numpy as np
from . import hm_calcs

paramlist = ['M', 'sigM', 'M0', 'M1', 'alpha']

paramdict = {'M': 0, 'sigM': 1, 'M0': 2, 'M1':3, 'alpha': 4}
param_bounds = {'M': (11, 14), 'sigM': (0, 2), 'M0': (11, 14),
                    'M1': (12, 16), 'alpha': (0.2, 2.5)}
default_priors = {'M': (None, None), 'sigM': (-0.5, 0.5), 'M0': (None, None),
                    'M1': (None, None), 'alpha': (1., 0.3)}

use_lognormal_prior = {'M': False, 'sigM': True, 'M0': False,
                    'M1': False, 'alpha': False}


def ln_normal(x, mean, sig):
	return np.log(1.0 / (np.sqrt(2*np.pi)*sig)) - 0.5 * ((x - mean) / sig) ** 2

def ln_lognormal(x, mean, sig):
	return np.log(1.0 / (np.sqrt(2*np.pi) * sig * x)) - 0.5 * ((np.log(x) - mean) / sig) ** 2



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

def parse_cf(cf, hm_ob):
	if 'theta_bins' in cf.keys():
		anglebins = cf['theta_bins']
		y = cf['w_theta']
		yerr = cf['w_err']
		modelprediction = hm_ob.get_binned_ang_cf(theta_bins=anglebins)

	elif 'rp_bins' in cf.keys():
		rp_bins = cf['rp_bins']
		y = cf['wp']
		yerr = cf['wp_err']
		modelprediction = hm_ob.get_binned_spatial_cf(radius_bins=rp_bins)
	else:
		print('inspect cf dict')
	return y, yerr, modelprediction


# log prior function
def ln_prior(hodparams):
	priorsum = 0
	for j in range(len(hodparams)):
		paramval = hodparams[j]
		if (paramval <= param_bounds[paramlist[j]][0]) or (paramval >= param_bounds[paramlist[j]][1]):
			priorsum -= np.inf
			return priorsum
		mu_j, sig_j = default_priors[paramlist[j]]
		if mu_j is not None:
			if use_lognormal_prior[paramlist[j]]:
				priorsum += ln_lognormal(paramval, mu_j, sig_j)
			else:
				priorsum += ln_normal(paramval, mu_j, sig_j)
	return priorsum



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
def ln_prob_cf(theta, cf, freeparam_ids, hmobj, xcorr_chain):
	hodparams = parse_params(theta, freeparam_ids)
	prior = ln_prior(hodparams)
	hodparams2 = None
	if xcorr_chain is not None:
		xcorr_link = np.random.choice(len(xcorr_chain), 1)
		hodparams2 = parse_params(xcorr_link, freeparam_ids)

	if prior > -np.inf:
		hmobj.set_powspec(hodparams=hodparams, hodparams2=hodparams2)
		# keep track of derived parameters like satellite fraction, effective bias, effective mass
		#derived = (hod_model.derived_parameters(zs, dndz, theta, modeltype))
		#derived = hmobj.hm.derived_parameters(dndz=dndz)

		# get model prediciton for given parameter set
		y, yerr, modelprediction = parse_cf(cf=cf, hm_ob=hmobj)

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

def clean_chain(chain, ids):
	"""
	Remove entries in chain which are outside prior bounds (a walker got "stuck")
	"""
	newchain = []
	for j in range(len(chain)):
		if np.isfinite(ln_prior(parse_params(chain[j], freeparam_ids=ids))):
			newchain.append(chain[j])
	return np.array(newchain)



def sample_cf_space(nwalkers, niter, cf, dndz, freeparam_ids, initial_params=None, pool=None, xcorr_chain=None):

	ndim = len(freeparam_ids)
	#blobs_dtype = [("f_sat", float), ("b_eff", float), ("m_eff", float)]
	#if ndim == 1:
	#	blobs_dtype = blobs_dtype[1:]

	halomod_obj = hm_calcs.halomodel(dndz)


	sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob_cf,
									args=[cf, freeparam_ids, halomod_obj, xcorr_chain],
	                                pool=pool)


	# start walkers near least squares fit position with random gaussian offsets
	pos = np.array(initial_params) + 2e-1 * np.random.normal(size=(sampler.nwalkers, sampler.ndim))
	pos = clean_chain(pos, freeparam_ids)

	sampler.run_mcmc(pos, niter, progress=True)


	flatchain = np.array(sampler.get_chain(flat=True))
	flatchain = clean_chain(flatchain, freeparam_ids)
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
	pos = clean_chain(pos, freeparam_ids)

	sampler.run_mcmc(pos, niter, progress=True)


	flatchain = sampler.get_chain(flat=True)
	flatchain = clean_chain(flatchain, freeparam_ids)
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
