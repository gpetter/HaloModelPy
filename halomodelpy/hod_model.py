import numpy as np
from colossus.lss import bias
from colossus.lss import mass_function
import astropy.units as u
from colossus.halo import concentration
from scipy import special
from functools import partial
import astropy.cosmology.units as cu
from . import params
from . import bias_tools
import camb

paramobj = params.param_obj()
col_cosmo = paramobj.col_cosmo
apcosmo = paramobj.apcosmo
param_keys = {'logMmin': 0, 'alpha': 1, 'logM1': 2}
mass_grid = paramobj.mass_space
k_grid = paramobj.k_space


# mass-concentration relation
def concentration_from_mass(masses, z):
	return concentration.concentration(masses, '200c', z, model='duffy08')


def lin_pk_z(zs, kspace, linpow='EH'):
	pkz = []
	if linpow == 'EH':
		for z in zs:
			pkz.append(col_cosmo.matterPowerSpectrum(kspace, z))
		return np.array(pkz)
	elif linpow == 'CAMB':
		cambpars = paramobj.cambpars
		cambpars.set_matter_power(redshifts=zs, kmax=2.)
		results = camb.get_results(cambpars)
		kh, z, pkz = results.get_matter_power_spectrum(minkh=np.min(kspace), maxkh=np.max(kspace), npoints=len(kspace))
		return np.array(pkz)
	#elif linpow == 'CAMB_interp':
		#camb_interpolator = paramobj.PK
		#for z in zs:
		#	pkz.append(camb_interpolator(z, kspace))
		#return np.array(pkz)
	else:
		return None


# take Fourier transform of NFW density profile
def transformed_nfw_profile(masses, z, analytic=True):
	import halomod
	from halomod.concentration import Duffy08
	from hmf import halos
	mdf = halos.mass_definitions.SOCritical()
	nfwprof = halomod.profiles.NFW(cm_relation=Duffy08(), mdef=mdf, z=z, cosmo=apcosmo)
	return nfwprof.u(k=k_grid, m=masses, norm='m', c=Duffy08().cm(masses, z=z))


# number density of halos at z per log mass interval
def halo_mass_function(masses, z):
	return mass_function.massFunction(masses, z, mdef='200c', model='tinker08', q_in='M', q_out='dndlnM') * (
		cu.littleh / u.Mpc) ** 3


# Zheng 2005 model
def three_param_hod(masses, logm_min, alpha, logm1):
	mmin = 10 ** logm_min
	m1 = 10 ** logm1
	# fix softening parameter
	sigma = paramobj.sigma_logM
	n_cen = 1 / 2. * (1 + special.erf(np.log10(masses/mmin) / sigma))
	n_sat = np.heaviside(masses - mmin, 1) * (((masses - mmin) / m1) ** alpha)
	n_sat[np.where(np.logical_not(np.isfinite(n_sat)))] = 0.

	return n_cen, n_sat


# a two parameter version of Zheng model where M_1 is fixed to be a constant multiple of M_min
def two_param_hod(masses, logm_min, alpha):
	# fix M1 to be a constant factor * M_0 (Georgakakis 2018)
	n_cen, n_sat = three_param_hod(masses, logm_min, alpha, logm_min + np.log10(paramobj.M1_over_M0))
	return n_cen, n_sat


def one_param_hod(masses, logm_min):
	ncen = np.zeros(len(masses))
	closeidx = np.abs(np.log10(masses) - logm_min).argmin()
	ncen[closeidx] = 1
	ncen[closeidx + 1] = 1
	nsat = np.zeros(len(masses))
	return ncen, nsat


# number of central AGN
def n_central(masses, params, modelname='zheng07'):
	if modelname == '1param':
		mean_ncen = one_param_hod(masses, params[param_keys['logMmin']])[0]
	elif modelname == '2param':
		mean_ncen = two_param_hod(masses, params[param_keys['logMmin']], params[param_keys['alpha']])[0]
	elif modelname == '3param':
		mean_ncen = three_param_hod(masses, params[param_keys['logMmin']], params[param_keys['alpha']],
		                            params[param_keys['logM1']])[0]

	elif modelname == 'dm':
		return np.zeros(len(masses))
	else:
		return None


	return mean_ncen


# number of satellites
def n_satellites(masses, params, modelname='zheng07'):
	if modelname == '1param':
		mean_nsat = one_param_hod(masses, params[param_keys['logMmin']])[1]
	elif modelname == '2param':
		mean_nsat = two_param_hod(masses, params[param_keys['logMmin']], params[param_keys['alpha']])[1]
	elif modelname == '3param':
		mean_nsat = three_param_hod(masses, params[param_keys['logMmin']], params[param_keys['alpha']],
		                            params[param_keys['logM1']])[1]
	elif modelname == 'dm':
		return np.zeros(len(masses))
	else:
		return None

	return mean_nsat


# sum of one and two halo terms
def hod_total(params, modeltype='zheng07'):
	return n_central(mass_grid, params, modeltype) + n_satellites(mass_grid, params, modeltype)


def fourier_variance(u_of_k_for_m_z, n_cen, n_sat):
	return np.transpose((2 * n_cen * n_sat * u_of_k_for_m_z +
	                                  ((n_cen*n_sat * u_of_k_for_m_z) ** 2)), axes=[1, 0, 2])


def one_halo_power_spectrum(k_grid, mass_grid, hmf_z, u_of_k_for_m_z, avg_dens_z, n_cen, n_sat):
	suppress_1h = True
	kstar = 1e-2

	integrand = hmf_z * np.transpose(n_cen * (2 * n_sat * u_of_k_for_m_z +
	                                  ((n_sat * u_of_k_for_m_z) ** 2)), axes=[1, 0, 2])

	integral = np.transpose((1 / avg_dens_z**2) * np.trapz(integrand, x=np.log(mass_grid)))
	if suppress_1h:
		# trick to suppress spurious 1-halo power at large scales
		integral *= ((k_grid / kstar) ** 4) / (1 + (k_grid / kstar) ** 4)

	return integral


# get halo bias as function of mass and redshift
# need this because you can't pass 2D grid of mass and redshift to colossus
def bias_m_z(mass_grid, zs):
	b_m_z = []
	for z in zs:
		b_m_z.append(bias.haloBias(mass_grid, model='tinker10', z=z, mdef='200c'))
	return np.array(b_m_z)

def halo_halo_power_spectrum(m1, m2, z):
	b1, b2 = bias.haloBias(m1, z, mdef='200c', model='tinker10'), bias.haloBias(m2, z, mdef='200c', model='tinker10')

	pk = col_cosmo.matterPowerSpectrum(k_grid, z) * (u.Mpc / cu.littleh) ** 3
	return np.outer(b1 * b2, pk)


# effective bias as function of redshift for HOD model
# this is used in the approximation that HOD power on large scales is just linearly biased wrt matter
# Eq 65 in Murray et al 2021
# otherwise this should also be a function of k, use b_eff_k_z
def b_eff_of_z(mass_grid, b_m_z, hmf_z, hod, avg_dens_z):
	return 1. / avg_dens_z * np.trapz(b_m_z * hod * hmf_z, x=np.log(mass_grid))


# effective bias as funciton of redshift and k
# scale dependent bias suppresses power at small scales in the 2halo term
# see e.g. Eq. 10 arxiv:1706.05422
def b_eff_k_z(mass_grid, b_m_z, hmf_z, ncen, nsat, ukm_z, avg_dens_z):

	bkz = 1. / avg_dens_z * np.trapz(b_m_z * np.transpose((ncen + nsat*ukm_z), axes=[1,0,2]) * hmf_z,
	                    x=np.log(mass_grid))

	return bkz


def two_halo_power_spectrum(b_arr, pk_z):
	return np.transpose((b_arr ** 2) * np.transpose(pk_z))


def log_effective_mass(hmf, hod, avg_dens, zs, dndz):
	meff_zs = 1. / avg_dens * np.trapz(mass_grid * hod * hmf, x=np.log(mass_grid))
	return np.log10(np.trapz(meff_zs * dndz, x=zs))


def satellite_fraction(hmf, n_sat, avg_dens, zs, dndz):
	fsat_zs = 1. / avg_dens * np.trapz(n_sat * hmf, x=np.log(mass_grid))
	return np.trapz(fsat_zs * dndz, x=zs)


def u_of_k_for_m_for_zs(masses, zs):
	uk_m_z = []
	for z in zs:
		uk_m_z.append(transformed_nfw_profile(masses, z))
	return np.array(uk_m_z)


def hmf_for_zs(masses, zs):
	hmfs = []
	for z in zs:
		hmfs.append(halo_mass_function(masses, z))
	return np.array(hmfs)


# integral of HOD over halo mass function gives average number density of AGN
def avg_dens_for_zs(hmf_zs, hod):
	return np.trapz(hmf_zs * hod, x=np.log(mass_grid), axis=1)


class halomod_workspace(object):
	def __init__(self, zs, linpow='EH'):
		self.zs = zs
		self.k_grid = k_grid
		self.mass_grid = mass_grid
		self.mass_def = '200c'
		self.bias_relation = partial(bias.haloBias, model='tinker10', mdef=self.mass_def)
		self.hmf_z = hmf_for_zs(masses=self.mass_grid, zs=zs)
		self.uk_m_z = u_of_k_for_m_for_zs(self.mass_grid, zs)
		self.linpk_z = lin_pk_z(zs=zs, kspace=self.k_grid, linpow=linpow)
		self.b_m_z = bias_m_z(self.mass_grid, self.zs)
		# HOD derived parameters
		self.ncen = None
		self.nsat = None
		self.ntot = None
		self.avgdens_z = None
		self.beff_z = None

	def set_hod(self, params, modeltype):
		self.ncen = n_central(masses=self.mass_grid, params=params, modelname=modeltype)
		self.nsat = n_satellites(masses=self.mass_grid, params=params, modelname=modeltype)
		self.ntot = self.ncen + self.nsat
		self.avgdens_z = avg_dens_for_zs(hmf_zs=self.hmf_z, hod=self.ntot)
		self.beff_z = b_eff_of_z(mass_grid=self.mass_grid, b_m_z=self.b_m_z, hmf_z=self.hmf_z, hod=self.ntot,
		                                            avg_dens_z=self.avgdens_z)
		#self.beff_k_z = b_eff_k_z(mass_grid=self.mass_grid, b_m_z=self.b_m_z, hmf_z=self.hmf_z, ncen=self.ncen,
		#                          nsat=self.nsat, ukm_z=self.uk_m_z, avg_dens_z=self.avgdens_z)

	def fourier_var(self):
		return fourier_variance(u_of_k_for_m_z=self.uk_m_z, n_cen=self.ncen, n_sat=self.nsat)

	def one_halo_power_z(self):
		return one_halo_power_spectrum(k_grid=self.k_grid, mass_grid=self.mass_grid, hmf_z=self.hmf_z,
		                                u_of_k_for_m_z=self.uk_m_z, avg_dens_z=self.avgdens_z, n_cen=self.ncen,
		                                n_sat=self.nsat)

	def two_halo_power_z(self):
		return two_halo_power_spectrum(b_arr=self.beff_z, pk_z=self.linpk_z)

	def hod_power_z(self, params, modeltype, get1h=True, get2h=True, smooth_transition_index=0.75):
		self.set_hod(params, modeltype)
		pow1h = self.one_halo_power_z()
		pow2h = self.two_halo_power_z()
		if get1h and get2h:
			powtot = (pow1h**(smooth_transition_index) + pow2h**smooth_transition_index)**(1./smooth_transition_index)
			return powtot
		if not get1h:
			return pow2h
		if not get2h:
			return pow1h

	def derived_parameters(self, dndz):
		beff = np.trapz(self.beff_z * dndz, x=self.zs)
		#meff = log_effective_mass(hmf=self.hmf_z, hod=self.ntot, avg_dens=self.avgdens_z, zs=self.zs, dndz=dndz)
		#meff = bias_tools.avg_bias_to_mass(beff, self.zs, dndz)
		f_sat = satellite_fraction(hmf=self.hmf_z, n_sat=self.nsat, avg_dens=self.avgdens_z, zs=self.zs, dndz=dndz)

		#return f_sat, beff, meff
		return f_sat, beff

def zheng_hod(params, param_ids, massgrid=np.logspace(11, 15, 100)):
	from . import mcmc
	logm, logsigm, logm0, logm1, alpha = mcmc.parse_params(params, param_ids)
	mmin = 10 ** logm
	m1 = 10 ** logm1
	# fix softening parameter
	sigma = logsigm
	n_cen = 1 / 2. * (1 + special.erf(np.log10(massgrid / mmin) / sigma))
	n_sat = np.heaviside(massgrid - mmin, 1) * (((massgrid - mmin) / m1) ** alpha)
	n_sat[np.where(np.logical_not(np.isfinite(n_sat)))] = 0.
	outdict = {}
	outdict['mgrid'] = np.log10(massgrid)
	outdict['ncen'] = n_cen
	outdict['nsat'] = n_sat
	outdict['hod'] = n_cen + n_sat

	return outdict
