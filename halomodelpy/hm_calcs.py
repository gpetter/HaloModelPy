import numpy as np
import astropy.units as u
import mcfit
from scipy.interpolate import interp1d
import astropy.constants as const
from . import hod_model
from colossus.cosmology import cosmology
import astropy.cosmology.units as cu
from functools import partial
from scipy import stats

col_cosmo = cosmology.setCosmology('planck18')
apcosmo = col_cosmo.toAstropy()


# ensure the redshift distribution is properly normalized
def norm_z_dist(dndz):
	return dndz[0], dndz[1] / np.trapz(dndz[1], x=dndz[0])


# eq from Liu et al. 2015
# return lensing kernel for CMB (if no source redshifts provided)
# or lensing kernel for galaxy weak lensing if source redshifts are provided
def lensing_kernel(lens_zs, chi_z_func=apcosmo.comoving_distance, H0=apcosmo.H0, om_0=apcosmo.Om0,
				   source_zs=None, cmb_z=1090.):
	prefactor = 3. / 2. * om_0 * ((H0 / const.c.to(u.km / u.s).value) ** 2) * (1 + lens_zs) * chi_z_func(lens_zs)
	# if no source redshifts, return CMB lensing kernel
	if source_zs is None:
		wk = prefactor * (chi_z_func(cmb_z) - chi_z_func(
			lens_zs)) / chi_z_func(cmb_z)
	# if source redshifts given,
	else:
		# calculate dn/dz for sources
		source_dist = np.histogram(source_zs, 200, density=True)
		dndz, source_bins = source_dist[0], source_dist[1]
		source_bin_centers = (source_bins[1:] + source_bins[:-1]) / 2

		integrals = []
		# for each redshift in lens list
		for lens_z in lens_zs:
			# integrate from z_lens to inf
			idxs = np.where(source_bin_centers > lens_z)

			if len(idxs[0]) > 1:
				new_source_zs, new_dndz = source_bin_centers[idxs], dndz[idxs]
				integrand = new_dndz * (chi_z_func(new_source_zs) - chi_z_func(lens_z)) \
							/ chi_z_func(new_source_zs)

				integrals.append(np.trapz(integrand, new_source_zs))
			else:
				integrals.append(0)
		wk = prefactor * np.array(integrals)

	return wk


def chi_z_func(zs, littleh_units=True):
	if littleh_units:
		return apcosmo.comoving_distance(zs).to(u.Mpc / cu.littleh, cu.with_H0(apcosmo.H0)).value
	return apcosmo.comoving_distance(zs).value


def Hubble_z(zs, littleh_units=True):
	if littleh_units:
		return apcosmo.H(zs).to((u.km * cu.littleh / u.s / u.Mpc), cu.with_H0(apcosmo.H0)).value
	else:
		return apcosmo.H(zs).value


# project power spectra as a function of redshift to an angular correlation function
# can do a cross-correlation function if second power spectrum and redshift distribution given
# the input power spectra should already be biased wrt dark matter if necessary
# make sure P(k), k, chi, H have uniform little h units
def pk_z_to_ang_cf(pk_z, dndz, thetas, k_grid, chi_zfunc, H_zfunc, pk_z_2=None, dndz_2=None):
	dndz = norm_z_dist(dndz)
	if dndz_2 is not None:
		dndz_2 = norm_z_dist(dndz_2)
	chi_z, H_z = chi_zfunc(dndz[0]), H_zfunc(dndz[0])

	# convert input thetas to radians from degrees
	thetas = (thetas * u.deg).to('radian').value

	# if doing a cross-correlation between two power spectra, (like two HOD P(k,z)'s),
	# we want to integrate the product of their square roots
	# and integrate over the product of their redshift distributions
	if pk_z_2 is not None:
		tot_pk_z = np.sqrt(pk_z) * np.sqrt(pk_z_2)
		dndz_prod = dndz[1] * dndz_2[1]
	else:
		tot_pk_z = pk_z
		dndz_prod = dndz[1] ** 2

	# Hankel transform power spectra at different redshifts to correlation functions which we will project to angular
	# space
	thetachis, dipomp_int = mcfit.Hankel(k_grid, lowring=True)(tot_pk_z, axis=1, extrap=True)

	# 2D grid of thetas * chi(z) to interpolate model power spectra onto
	input_theta_chis = np.outer(chi_z, thetas)

	# for each redshift, chi(z), interpolate the result of the above integral onto theta*chi(z) grid
	interped_dipomp = []
	for j in range(len(dndz[0])):
		# trick for interpolation for function varying in log space (propto r^-2)
		# multiply result of Hankel transform by theta*chi, this makes it pretty smooth in linear space
		# then divide it back out at end
		flatpower = dipomp_int[j] * thetachis
		interped_dipomp.append(interp1d(thetachis, flatpower)(input_theta_chis[j]) /
							   input_theta_chis[j])
	interped_dipomp = np.array(interped_dipomp)

	dz_d_chi = (H_z / const.c.to(u.km / u.s).value)
	# product of redshift distributions, and dz/dchi
	differentials = dz_d_chi * dndz_prod

	# integrate over redshift kernel to get w(theta) a la Dipompeo+2017
	return 1 / (2 * np.pi) * np.trapz(differentials * np.transpose(interped_dipomp), x=dndz[0], axis=1)


# Fourier transform input power spectra to correlation functions, convert to projected CFs wp(rp) if desired,
# and project to observable for given redshift distribution(s)
def pk_z_to_xi_r(pk_z, dndz, radii, k_grid, pk_z_2=None, dndz_2=None, projected=False):
	dndz = norm_z_dist(dndz)
	if dndz_2 is not None:
		dndz_2 = norm_z_dist(dndz_2)
	import abel

	# if doing a cross-correlation between two power spectra, (like two HOD P(k,z)'s),
	# we want to integrate the product of their square roots
	# and integrate over the product of their redshift distributions
	if pk_z_2 is not None:
		tot_pk_z = np.sqrt(pk_z) * np.sqrt(pk_z_2)
		dndz_prod = dndz[1] * dndz_2[1]
	else:
		tot_pk_z = pk_z
		dndz_prod = dndz[1] ** 2

	rgrid, xis = mcfit.P2xi(k_grid, lowring=True)(tot_pk_z, axis=1, extrap=True)

	if projected:
		xis = np.array(abel.direct.direct_transform(xis, r=rgrid, direction='forward', backend='python'))
	# am i sure this is okay for the projected correlation function? am i actually getting it at r_p?

	# trick to make interpolation work for logarithmically varying xi (propto r^-2)
	# multiply xi by r to make smooth in linear space, then divide r back out at end
	interpedxis = interp1d(rgrid, xis * rgrid)(radii) / radii

	return np.trapz(dndz_prod * np.transpose(interpedxis), x=dndz[0], axis=1)


# get cross spectrm C_ell between galaxy overdensity and gravitational lensing
def c_ell_kappa_g(pk_z, dndz, ls, k_grid, chi_z_func, H_z_func, lin_pk_z):
	dndz = norm_z_dist(dndz)
	lenskern = lensing_kernel(lens_zs=dndz[0], chi_z_func=chi_z_func, H0=H_z_func(0))
	qsokern = H_z_func(dndz[0]) / const.c.to(u.km / u.s).value * dndz[1]

	integrand = (const.c.to(u.km / u.s).value * lenskern * qsokern / ((chi_z_func(dndz[0]) ** 2) * H_z_func(dndz[0])))
	ks = np.outer(1. / chi_z_func(dndz[0]), (ls + 1 / 2.))

	ps_at_ks = []
	pk_z = np.sqrt(pk_z) * np.sqrt(lin_pk_z)

	for j in range(len(dndz[0])):
		ps_at_ks.append(np.interp(ks[j], k_grid, pk_z[j]))

	ps_at_ks = np.array(ps_at_ks)

	integrand = integrand[:, None] * ps_at_ks

	return np.trapz(integrand, dndz[0], axis=0)


# compute statistics for dark matter or tracers linearly biased against it
class halomodel(object):

	def __init__(self, zs, littleh_units=True):
		self.zs = zs
		self.hm = hod_model.halomod_workspace(zs=self.zs)
		self.pk_z = self.hm.linpk_z
		self.k_grid = self.hm.k_grid
		self.chizfunc = partial(chi_z_func, littleh_units=littleh_units)
		self.hzfunc = partial(Hubble_z, littleh_units=littleh_units)
		# second power spectrum in case of cross-correlation
		self.hm2 = None
		self.pk_z_2 = None

	# reset the power spectrum according to an HOD if provided, or an effective mass-biased spectrum
	def set_powspec(self, hodparams=None, modeltype=None, hodparams2=None, log_meff=None, log_meff_2=None,
					get1h=True, get2h=True):

		if hodparams is not None:
			self.hm.set_hod(params=hodparams, modeltype=modeltype)
			self.pk_z = self.hm.hod_power_z(get1h=get1h, get2h=get2h)
		if hodparams2 is not None:
			self.hm2 = hod_model.halomod_workspace(zs=self.zs)
			self.hm2.set_hod(params=hodparams2, modeltype=modeltype)
			self.pk_z_2 = self.hm2.hod_power_z(get1h=get1h, get2h=get2h)
		# if considering a population of halos with effective mass M rather than full HOD
		if log_meff is not None:
			bz = self.hm.bias_relation(M=10 ** log_meff, z=self.zs)
			self.pk_z = (bz ** 2)[:, None] * self.hm.linpk_z
		if log_meff_2 is not None:
			self.pk_z_2 = self.hm.linpk_z
			bz = self.hm.bias_relation(M=10 ** log_meff_2, z=self.zs)
			self.pk_z_2 = (bz ** 2)[:, None] * self.pk_z_2

	def get_ang_cf(self, dndz, thetas, dndz_2=None):
		return pk_z_to_ang_cf(pk_z=self.pk_z, dndz=dndz, thetas=thetas, k_grid=self.k_grid,
							  chi_zfunc=self.chizfunc, H_zfunc=self.hzfunc, pk_z_2=self.pk_z_2, dndz_2=dndz_2)

	def get_binned_ang_cf(self, dndz, theta_bins, dndz_2=None):
		thetagrid = np.logspace(-3, 1, 100)
		wtheta = self.get_ang_cf(dndz=dndz, thetas=thetagrid, dndz_2=dndz_2)
		return stats.binned_statistic(thetagrid, wtheta, statistic='mean', bins=theta_bins)[0]

	def get_spatial_cf(self, dndz, radii, dndz_2=None, projected=False):
		return pk_z_to_xi_r(pk_z=self.pk_z, dndz=dndz, radii=radii, k_grid=self.k_grid, pk_z_2=self.pk_z_2,
							dndz_2=dndz_2, projected=projected)

	def get_c_ell_kg(self, dndz, ls):
		return c_ell_kappa_g(pk_z=self.pk_z, dndz=dndz, ls=ls, k_grid=self.k_grid, chi_z_func=self.chizfunc,
							 H_z_func=self.hzfunc, lin_pk_z=self.hm.linpk_z)
