
import numpy as np
from colossus.cosmology import cosmology
cosmo = cosmology.setCosmology('planck18')
from colossus.lss import bias
from colossus.lss import mass_function

def mass2bias(logmh, z):
	return bias.haloBias(M=10**logmh, z=z, mdef='200c', model='tinker10')

def bias2mass(inputbias, z):
	masses = np.logspace(10, 15, 1000)
	biases_from_masses = bias.haloBias(M=masses, z=z, mdef='200c', model='tinker10')
	return np.interp(inputbias, biases_from_masses, np.log10(masses))


# take a characteristic halo mass and calculate the resulting average bias over a given redshift distribution
def mass_to_avg_bias(log_m, zs, dndz, log_merr=None):

	bh = bias.haloBias(M=10 ** log_m, z=zs, mdef='200c', model='tinker10')

	avg_bh = np.trapz(bh*dndz, x=zs)
	if log_merr is not None:
		bh_plus = bias.haloBias(M=10 ** (log_m+log_merr[0]), z=zs, mdef='200c', model='tinker10')
		bh_minus = bias.haloBias(M=10 ** (log_m-log_merr[1]), z=zs, mdef='200c', model='tinker10')
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
		mfunc = mass_function.massFunction(massgrid, z, mdef='200c', model='tinker08', q_out='dndlnM')
		bm_z = bias.haloBias(M=massgrid, z=z, mdef='200c', model='tinker10')
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