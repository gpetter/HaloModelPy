
import numpy as np
from colossus.cosmology import cosmology
cosmo = cosmology.setCosmology('planck18')
from colossus.lss import bias



def bias_to_mass(inputbias, z):
	masses = np.logspace(10, 14, 1000)
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