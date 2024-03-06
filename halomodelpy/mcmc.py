import emcee
import numpy as np
from . import hm_calcs
#

#paramdict = {'M': 0, 'sigM': 1, 'M0': 2, 'M1':3, 'alpha': 4}
#param_bounds = {'M': (11, 14), 'sigM': (0, 2), 'M0': (11, 14),
#                    'M1': (12, 16), 'alpha': (0.2, 2.5)}

#default_priors = {'M': (13., 2.), 'sigM': (0.5, 0.5), 'M0': (None, None),
#					  'M1': (14., 2.), 'alpha': (1., 1)}


def ln_normal(x, mean, sig):
	return np.log(1.0 / (np.sqrt(2 * np.pi) * sig)) - 0.5 * ((x - mean) / sig) ** 2


def ln_lognormal(x, mean, sig):
	return np.log(1.0 / (np.sqrt(2 * np.pi) * sig * x)) - 0.5 * ((np.log(x) - mean) / sig) ** 2


class zhengHODsampler(object):

	def __init__(self):
		self.paramlist = ['M', 'sigM', 'M0', 'M1', 'alpha']
		self.paramdict = {'M': 0, 'sigM': 1, 'M0': 2, 'M1': 3, 'alpha': 4}
		# default priors
		self.prior_params = {'M': (13., 2.), 'sigM': (0.5, 0.5), 'M0': (None, None),
							 'M1': (14., 2.), 'alpha': (1., 1)}
		self.param_bounds = {'M': (11, 14), 'sigM': (0, 2), 'M0': (11, 14), 'M1': (12, 16), 'alpha': (0.2, 2.5)}
		self.default_hod = np.array([12.5, 1e-3, 12.5, 13.5, 1.])
		self.use_lognormal_prior = {'M': False, 'sigM': False, 'M0': False,
									'M1': False, 'alpha': False}

	def update_priors(self, priordict):
		for priorid in priordict:
			self.prior_params[priorid] = priordict[priorid]

	def parse_params(self, theta, freeparam_ids):

		hodparams = np.copy(self.default_hod)
		for j in range(len(freeparam_ids)):
			hodparams[self.paramdict[freeparam_ids[j]]] = theta[j]
		# if M0 not a free parameter, set equal to M_min
		if 'M0' not in freeparam_ids:
			hodparams[self.paramdict['M0']] = theta[0]
		# if M1 not a free parameter, set to 10x M_min
		if 'M1' not in freeparam_ids:
			hodparams[self.paramdict['M1']] = 1. + theta[0]
		return hodparams

	def parse_cf(self, cf, hm_ob):
		"""
		Choose whether modeling either an angular or spatial correlation function
		:param cf:
		:param hm_ob:
		:return:
		"""
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
	def ln_prior(self, hodparams):
		priorsum = 0
		for j in range(len(hodparams)):
			paramval = hodparams[j]
			# prior of zero if outside bounds
			if (paramval <= self.param_bounds[self.paramlist[j]][0]) or \
					(paramval >= self.param_bounds[self.paramlist[j]][1]):
				priorsum -= np.inf
				return priorsum

			mu_j, sig_j = self.prior_params[self.paramlist[j]]

			if mu_j is not None:
				if self.use_lognormal_prior[self.paramlist[j]]:
					priorsum += ln_lognormal(paramval, mu_j, sig_j)
				else:
					priorsum += ln_normal(paramval, mu_j, sig_j)
		return priorsum

	# log likelihood function
	def ln_likelihood(self, residual, yerr):

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
	def ln_prob_cf(self, theta, cf, freeparam_ids, hmobj, xcorr_chain=None):
		hodparams = self.parse_params(theta, freeparam_ids)
		prior = self.ln_prior(hodparams)
		if xcorr_chain is not None:
			# randomly draw a link from the cross-correlation sample chain
			xcorr_link = xcorr_chain[np.random.choice(len(xcorr_chain)), 1]
			hodparams2 = self.parse_params(theta=theta, freeparam_ids=freeparam_ids)
		else:
			hodparams2 = None


		if prior > -np.inf:
			hmobj.set_powspec(hodparams=hodparams, hodparams2=hodparams2)

			# get model prediciton for given parameter set
			y, yerr, modelprediction = self.parse_cf(cf=cf, hm_ob=hmobj)

			# residual is data - model
			residual = y - modelprediction

			likely = self.ln_likelihood(residual, yerr)
			prob = prior + likely

			# return log_prob, along with derived parameters for this parameter set
			# return (prob,) + derived
			return prob
		else:
			return prior

	# log probability is prior plus likelihood
	def ln_prob_lens(self, theta, xcorr, freeparam_ids, hmobj):
		ell_bins = xcorr['ell_bins']
		y = xcorr['cl']
		yerr = xcorr['cl_err']
		hodparams = self.parse_params(theta, freeparam_ids)
		prior = self.ln_prior(hodparams)
		if prior > -np.inf:
			hmobj.set_powspec(hodparams=hodparams)
			# keep track of derived parameters like satellite fraction, effective bias, effective mass
			# derived = (hod_model.derived_parameters(zs, dndz, theta, modeltype))
			# derived = hmobj.hm.derived_parameters(dndz=dndz)

			# get model prediciton for given parameter set
			modelprediction = hmobj.get_binned_c_ell_kg(ell_bins)

			# residual is data - model
			residual = y - modelprediction

			likely = self.ln_likelihood(residual, yerr)
			prob = prior + likely

			# return log_prob, along with derived parameters for this parameter set
			# return (prob,) + derived
			return prob
		else:
			return prior

	def clean_chain(self, chain, ids):
		"""
		Remove entries in chain which are outside prior bounds (a walker got "stuck")
		"""
		newchain = []
		for j in range(len(chain)):
			if np.isfinite(self.ln_prior(self.parse_params(chain[j], freeparam_ids=ids))):
				newchain.append(chain[j])
		return np.array(newchain)

	def sample_cf_space(self, nwalkers, niter, cf, dndz, freeparam_ids, initial_params=None, pool=None):

		ndim = len(freeparam_ids)
		halomod_obj = hm_calcs.halomodel(dndz)

		sampler = emcee.EnsembleSampler(nwalkers, ndim, self.ln_prob_cf,
										args=[cf, freeparam_ids, halomod_obj],
										pool=pool)
		# start walkers near least squares fit position with random gaussian offsets
		# generate 3x more than you need, as some will probably scatter outside acceptable parameter range
		pos = np.array(initial_params) + 2e-1 * np.random.normal(size=(3 * sampler.nwalkers, sampler.ndim))
		# remove walkers started outside parameter range
		pos = self.clean_chain(pos, freeparam_ids)
		pos = pos[np.random.choice(len(pos), sampler.nwalkers, replace=False)]

		sampler.run_mcmc(pos, niter, progress=True)

		flatchain = np.array(sampler.get_chain(flat=True))
		flatchain = self.clean_chain(flatchain, freeparam_ids)

		return flatchain

	def sample_lens_space(self, nwalkers, niter, xcorr, dndz, freeparam_ids, initial_params=None, pool=None):
		ndim = len(freeparam_ids)

		halomod_obj = hm_calcs.halomodel(dndz)

		sampler = emcee.EnsembleSampler(nwalkers, ndim, self.ln_prob_lens, args=[xcorr, freeparam_ids, halomod_obj],
										pool=pool)

		# start walkers near least squares fit position with random gaussian offsets
		# generate 3x more than you need, as some will probably scatter outside acceptable parameter range
		pos = np.array(initial_params) + 2e-1 * np.random.normal(size=(3 * sampler.nwalkers, sampler.ndim))
		# remove walkers started outside parameter range
		pos = self.clean_chain(pos, freeparam_ids)
		pos = pos[np.random.choice(len(pos), sampler.nwalkers, replace=False)]

		sampler.run_mcmc(pos, niter, progress=True)

		flatchain = sampler.get_chain(flat=True)
		flatchain = self.clean_chain(flatchain, freeparam_ids)
		return flatchain

	def sample_xcf_space(self, nwalkers, niter, xcf, dndz_x, dndz_ref, ref_hod_chain, freeparam_ids,
						 initial_params=None, pool=None):

		ndim = len(freeparam_ids)
		halomod_obj = hm_calcs.halomodel(dndz1=dndz_x, dndz2=dndz_ref)

		sampler = emcee.EnsembleSampler(nwalkers, ndim, self.ln_prob_cf,
										args=[xcf, freeparam_ids, halomod_obj, ref_hod_chain], pool=pool)

		# start walkers near least squares fit position with random gaussian offsets
		# generate 3x more than you need, as some will probably scatter outside acceptable parameter range
		pos = np.array(initial_params) + 2e-1 * np.random.normal(size=(3 * sampler.nwalkers, sampler.ndim))
		# remove walkers started outside parameter range
		pos = self.clean_chain(pos, freeparam_ids)
		pos = pos[np.random.choice(len(pos), sampler.nwalkers, replace=False)]

		sampler.run_mcmc(pos, niter, progress=True)

		flatchain = np.array(sampler.get_chain(flat=True))
		flatchain = self.clean_chain(flatchain, freeparam_ids)

		return flatchain