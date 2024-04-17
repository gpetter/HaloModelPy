import numpy as np
from colossus.lss import mass_function
from scipy import special
from . import params
from . import cosmo
paramobj = params.param_obj()
mdef = paramobj.mass_def
bias_relation = paramobj.bias_relation


def mass2bias(logmh, z):
	"""
	Convert halo mass to bias at redshift
	:param logmh: log10(Halo mass) (hubble units)
	:param z: redshift
	:return: bias
	"""
	return bias_relation(M=10**logmh, z=z)


def bias2mass(inputbias, z):
	"""
	Convert bias to halo mass at redshift
	:param inputbias: bias
	:param z: redshift
	:return: log10(halo mass) (hubble units)
	"""
	# grid of masses for interpolation
	masses = np.linspace(10, 15, 1000)
	# convert to biases
	biases_from_masses = mass2bias(masses, z)
	# interpolate in log mass space
	return np.interp(inputbias, biases_from_masses, masses)


def mass_to_avg_bias(log_mh, zs, dndz, log_merr=None):
	"""
	take a characteristic halo mass and calculate the resulting effective bias over a given redshift distribution
	assuming the mass is constant with z
	:param log_mh: log10(Halo mass) (hubble units)
	:param zs: redshifts array
	:param dndz: redshift distribution array
	:param log_merr: optionally, error in log(mass)
	:return: effective bias
	"""

	# bias for each z
	bh = mass2bias(logmh=log_mh, z=zs)

	# effective bias is integral over b*dn/dz
	avg_bh = np.trapz(bh*dndz, x=zs)
	# if wanting error in bias
	if log_merr is not None:
		bh_plus = bias_relation(M=10 ** (log_mh+log_merr[0]), z=zs)
		bh_minus = bias_relation(M=10 ** (log_mh-log_merr[1]), z=zs)
		avg_bh_err = np.mean([bh_plus - avg_bh, avg_bh - bh_minus])
		return avg_bh, avg_bh_err

	return avg_bh



def avg_bias_to_mass(input_bias, zs, dndz, berr=0):
	"""
	take a bias measured over a redshift distribution and calculate which characteristic halo mass would
	result in the measured bias
	:param input_bias:
	:param zs: redshifts array
	:param dndz: redshift distribution array
	:param berr: optional bias uncertainty
	:return:
	"""

	# mass grid
	masses = np.log10(np.logspace(10, 15, 500))
	b_avg = []
	# get effective bias over dn/dz for each mass in grid
	for mass in masses:
		b_avg.append(mass_to_avg_bias(mass, zs, dndz))

	# if bias error passed
	if berr > 0:
		upmass = np.interp(input_bias+berr, b_avg, masses)
		lomass = np.interp(input_bias-berr, b_avg, masses)
		mass = np.interp(input_bias, b_avg, masses)
		return mass, mass-lomass, upmass-mass
	else:
		# interpolate grid to get effective mass
		return np.interp(input_bias, b_avg, masses)



def minmass_to_bias_z(log_minmass, zs):
	"""
	calculate a bias corresponding to a minimum halo mass required to host a tracer
	e.g. Petter et al 2023 Eq. 19
	Involves integral over halo mass function
	:param log_minmass: minimum mass in log Hubble units
	:param zs: redshift array
	:return: b(z) corresponding to minimum mass
	"""
	zs = np.atleast_1d(zs)
	# grid from given minimum mass to high mass where HMF --> 0
	massgrid = np.logspace(log_minmass, 15, 100)
	beffs = []
	# for each redshift
	for z in zs:
		# effective bias at redshift is integral of HMF(M, z)*bias(M, z)*dM
		mfunc = paramobj.hmf(massgrid, z)
		bm_z = bias_relation(M=massgrid, z=z)
		beffs.append(np.trapz(bm_z * mfunc, x=np.log(massgrid)) / np.trapz(mfunc, x=np.log(massgrid)))
	return np.array(beffs)


def minmass_to_bias(dndz, log_minmass):
	"""
	Compute effective bias integrated over redshift distribution from minimum halo mass
	:param dndz: tuple (zs, dn/dz)
	:param log_minmass: minimum mass in log Hubble units
	:return:
	"""

	zs, dndzs = dndz
	beffs = minmass_to_bias_z(log_minmass, zs)

	# average over redshifts is integral over dN/dz
	return np.trapz(np.array(beffs) * np.array(dndzs), x=zs)


#
def bias_to_minmass(dndz, bias, minmass_grid=np.linspace(11., 14.5, 100)):
	"""
	go from effective bias to minimum host halo mass by inverting above function
	:param dndz: go from effective bias to minimum host halo mass by inverting above function
	:param bias: effective bias
	:param minmass_grid: grid of minimum masses in log Hubble units
	:return: minimum mass corresponding to effective bias
	"""

	biases_for_minmasses = []
	for j in range(len(minmass_grid)):
		biases_for_minmasses.append(minmass_to_bias(dndz, minmass_grid[j]))
	return np.interp(bias, biases_for_minmasses, minmass_grid)

def avg_bias2min_mass(dndz, b, berr=0):
	"""

	:param dndz: tuple (zs, dndz)
	:param b: bias
	:param sigma: smoothing parameter
	:param berr: Optional bias error
	:return:
	"""
	mmin = bias_to_minmass(dndz=dndz, bias=b)
	mupp, mlo = 0, 0
	if berr > 0:
		mupp = bias_to_minmass(dndz=dndz, bias=(b+berr)) - mmin
		mlo = mmin - bias_to_minmass(dndz=dndz, bias=(b-berr))
	return mmin, mlo, mupp

def ncen_zheng(logMmin, sigma):
	"""
	Central occupation function from Zheng+07, a softened step function
	:param logMmin:
	:param sigma:
	:return:
	"""
	return 1 / 2. * (1 + special.erf(np.log10(paramobj.mass_space / (10**logMmin)) / sigma))

def mass_transition2bias_z(logMmin, sigma, zs):
	"""
	Predict the b(z) corresponding to a (softened) step function HOD
	:param logMmin: minimum mass to host galaxy, log Hubble units
	:param sigma: smoothing parameter
	:param zs: redshift array
	:return: b(z)
	"""
	zs = np.atleast_1d(zs)
	hod = ncen_zheng(logMmin, sigma)
	beffs = []
	for z in zs:
		hmf = cosmo.hmf_z(logM_hubble=np.log10(paramobj.mass_space), z=z)
		bm_z = bias_relation(M=paramobj.mass_space, z=z)
		beffs.append(np.trapz(bm_z * hmf * hod, x=np.log(paramobj.mass_space)) /
					np.trapz(hmf * hod, x=np.log(paramobj.mass_space)))
	return np.array(beffs)

def mass_transition2bias(dndz, logMmin, sigma):
	"""
	Predict the effective bias for a (softened) step function HOD over a redshift distribution
	:param dndz: tuple (zs, dndz)
	:param logMmin: minimum mass to host galaxy, log Hubble units
	:param sigma: smoothing parameter
	:return: effective bias
	"""
	beffs = mass_transition2bias_z(logMmin=logMmin, sigma=sigma, zs=dndz[0])
	# average over redshifts is integral over dN/dz
	return np.trapz(np.array(beffs) * np.array(dndz[1]), x=dndz[0])

def avg_bias2mass_transition(dndz, b, sigma, berr=0.):
	"""
	Invert above function, go from an effective bias to a transition HOD mass
	:param dndz: tuple (zs, dndz)
	:param b: bias
	:param sigma: smoothing parameter
	:param berr: Optional bias error
	:return:
	"""
	ms = np.linspace(11., 14.5, 100)
	bs = []
	for m in ms:
		bs.append(mass_transition2bias(dndz=dndz, logMmin=m, sigma=sigma))
	mtrans = np.interp(b, bs, ms)
	mupp, mlo = 0, 0
	if berr > 0:
		mupp = np.interp(b + berr, bs, ms) - mtrans
		mlo = mtrans - np.interp(b - berr, bs, ms)
	return mtrans, mlo, mupp



def qso_bias_for_z(paper, zs, dndz=None, n_draws=100):
	"""
	calculate bias predicted by parameterizations of quasar bias as function of redshift given by
	Croom et al 2005 or Laurent et al. 2017
	:param paper: 'laurent' or 'croom' for Laurent et al. 2017 or Croom et al. 2005
	:param zs: redshift array
	:param dndz: redshift distribution
	:param n_draws: Number of random draws to estimate uncertainty
	:return: bias as a function of redshift from the paper
	"""

	if paper == 'laurent':
		a0, a1, a2 = 2.393, 0.278, 6.565
		sig_a0, sig_a1 = 0.042, 0.018
	elif paper == 'croom':
		a0, a1, a2 = 0.53, 0.289, 0
		sig_a0, sig_a1 = 0.19, 0.035
	else:
		return 'Invalid paper name. Use croom for Croom+05 or laurent for Laurent+17'

	# form of Croom and Laurent parameterization of bias with redshift
	lit_bias = a0 + a1 * ((1+zs) ** 2 - a2)

	# same as above but randomly drawn from error in parameterization n_draws times
	bias_draws = np.repeat(np.random.normal(a0, sig_a0, n_draws)[None, :], len(zs), axis=0) \
	             + np.outer(((1+zs) ** 2 - a2), np.random.normal(a1, sig_a1, n_draws))

	# if a redshift distribution given, average over using dn/dz as weight
	if dndz is not None:
		avg_b = np.average(lit_bias, weights=dndz)

		avg_b_draws = np.average(bias_draws, weights=dndz, axis=0)

		avg_b_std = np.std(avg_b_draws, axis=0)
		return avg_b, avg_b_std

	# if no redshift distribution given, just return the bias at each z, and the uncertainty at each z
	else:
		b_std = np.std(bias_draws, axis=1)
		return lit_bias, b_std


def qso_mass_for_z(paper, zs, dndz=None, n_draws=100):
	"""
	convert bias predicted from qso_bias_for_z into effective halo mass
	:return:
	"""
	b, berr = qso_bias_for_z(paper, zs, dndz, n_draws)
	m, mhi, mlo = [], [], []
	for j in range(len(zs)):
		m.append(bias2mass(b[j], zs[j]))
		mhi.append(bias2mass(b[j] + berr[j], zs[j]))
		mlo.append(bias2mass(b[j] - berr[j], zs[j]))
	return m, mhi, mlo