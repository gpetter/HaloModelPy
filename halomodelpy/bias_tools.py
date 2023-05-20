import numpy as np
from colossus.lss import mass_function
from . import params
paramobj = params.param_obj()
mdef = paramobj.mass_def
bias_relation = paramobj.bias_relation


def mass2bias(logmh, z):
	return bias_relation(M=10**logmh, z=z)


def bias2mass(inputbias, z):
	masses = np.logspace(10, 15, 1000)
	biases_from_masses = bias_relation(M=masses, z=z)
	return np.interp(inputbias, biases_from_masses, np.log10(masses))


# take a characteristic halo mass and calculate the resulting average bias over a given redshift distribution
def mass_to_avg_bias(log_m, zs, dndz, log_merr=None):

	bh = bias_relation(M=10 ** log_m, z=zs)

	avg_bh = np.trapz(bh*dndz, x=zs)
	if log_merr is not None:
		bh_plus = bias_relation(M=10 ** (log_m+log_merr[0]), z=zs)
		bh_minus = bias_relation(M=10 ** (log_m-log_merr[1]), z=zs)
		avg_bh_err = np.mean([bh_plus - avg_bh, avg_bh - bh_minus])
		return avg_bh, avg_bh_err

	return avg_bh


# take a bias measured over a redshift distribution and calculate which characteristic halo mass would
# result in the measured bias
def avg_bias_to_mass(input_bias, zs, dndz, berr=0):


	masses = np.log10(np.logspace(10, 15, 500))
	b_avg = []
	for mass in masses:
		b_avg.append(mass_to_avg_bias(mass, zs, dndz))

	if berr > 0:
		upmass = np.interp(input_bias+berr, b_avg, masses)
		lomass = np.interp(input_bias-berr, b_avg, masses)
		mass = np.interp(input_bias, b_avg, masses)
		return mass, mass-lomass, upmass-mass
	else:
		return np.interp(input_bias, b_avg, masses)


# calculate a bias corresponding to a minimum halo mass required to host a tracer
# e.g. Petter et al 2023 Eq. 19
def minmass_to_bias_z(log_minmass, zs):
	# grid from given minimum mass to high mass where HMF --> 0
	massgrid = np.logspace(log_minmass, 15, 100)
	beffs = []
	# for each redshift
	for z in zs:
		# effective bias at redshift is integral of HMF(M, z)*bias(M, z)*dM
		# TODO retrieve hmf from params.py
		mfunc = mass_function.massFunction(massgrid, z, mdef='200c', model='tinker08', q_out='dndlnM')
		bm_z = bias_relation(M=massgrid, z=z)
		beffs.append(np.trapz(bm_z * mfunc, x=np.log(massgrid)) / np.trapz(mfunc, x=np.log(massgrid)))
	return np.array(beffs)


def minmass_to_bias(dndz, log_minmass):

	zs, dndzs = dndz
	beffs = minmass_to_bias_z(log_minmass, zs)

	# average over redshifts is integral over dN/dz
	return np.trapz(np.array(beffs) * np.array(dndzs), x=zs)


# go from effective bias to minimum host halo mass by inverting above function
def bias_to_minmass(dndz, bias, minmass_grid=np.linspace(12., 14.5, 50)):

	biases_for_minmasses = []
	for j in range(len(minmass_grid)):
		biases_for_minmasses.append(minmass_to_bias(dndz, minmass_grid[j]))
	return np.interp(bias, biases_for_minmasses, minmass_grid)


# calculate bias predicted by parameterizations of quasar bias as function of redshift given by
# Croom et al 2005 or Laurent et al. 2017
def qso_bias_for_z(paper, zs, dndz=None, n_draws=100):

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
	b, berr = qso_bias_for_z(paper, zs, dndz, n_draws)
	m, mhi, mlo = [], [], []
	for j in range(len(zs)):
		m.append(bias2mass(b[j], zs[j]))
		mhi.append(bias2mass(b[j] + berr[j], zs[j]))
		mlo.append(bias2mass(b[j] - berr[j], zs[j]))
	return m, mhi, mlo