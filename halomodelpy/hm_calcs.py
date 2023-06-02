import numpy as np
import astropy.units as u
import mcfit
from scipy.interpolate import interp1d
import astropy.constants as const
import astropy.cosmology.units as cu
from functools import partial
from scipy import stats
from scipy.special import j0, legendre
from . import redshift_helper
from . import params
from . import bias_tools
from . import ccl_tools
paramobj = params.param_obj()
col_cosmo = paramobj.col_cosmo
apcosmo = paramobj.apcosmo


# eq from Liu et al. 2015
# return lensing kernel for CMB (if no source redshifts provided)
# or lensing kernel for galaxy weak lensing if source redshifts are provided
def lensing_kernel(lens_zs, chi_z_func=apcosmo.comoving_distance, H0=apcosmo.H0, om_0=apcosmo.Om0,
				   source_zs=None):
	prefactor = 3. / 2. * om_0 * ((H0 / const.c.to(u.km / u.s).value) ** 2) * (1 + lens_zs) * chi_z_func(lens_zs)
	# if no source redshifts, return CMB lensing kernel
	if source_zs is None:
		wk = prefactor * (chi_z_func(paramobj.cmb_z) - chi_z_func(
			lens_zs)) / chi_z_func(paramobj.cmb_z)
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

def beta_param(bs, zs):
	return col_cosmo.Om(zs) ** 0.56 / bs


def hamilton_j_interp(xi, sgrid, ss, n):
	jx = []
	for s in ss:
		maxidx = np.argmin(np.abs(s - sgrid))
		jx.append(np.trapz(xi[:, :maxidx] * sgrid[:maxidx] ** (n-1), x=sgrid[:maxidx]))
	lin_interp = interp1d(np.log10(ss), np.log10(jx), axis=0)
	return lambda zz: np.transpose(np.power(10.0, lin_interp(np.log10(zz))))

def xi_monopole(xi, beta):
	return (1 + 2/3 * beta + 1/5 * beta ** 2) * xi

def xi_quadrupole(s, xi, beta, j3, bz):
	return (4/3 * beta + 4/7 * beta ** 2) * (xi - 3 * (bz ** 2 * j3(s)) / s ** 3)

def xi_hexadecapole(s, xi, beta, j3, j5, bz):
	return 8/35 * beta ** 2 * (xi + 15/(2 * s ** 3) * (bz ** 2 * j3(s)) - 35/(2 * s ** 5) * (bz ** 2 * j5(s)))




# project power spectra as a function of redshift to an angular correlation function
# can do a cross-correlation function if second power spectrum and redshift distribution given
# the input power spectra should already be biased wrt dark matter if necessary
# make sure P(k), k, chi, H have uniform little h units
def pk_z_to_ang_cf(pk_z, dndz, thetas, k_grid, chi_zfunc, H_zfunc):
	chi_z, H_z = chi_zfunc(dndz[0]), H_zfunc(dndz[0])

	# convert input thetas to radians from degrees
	thetas = (thetas * u.deg).to('radian').value


	# Hankel transform power spectra at different redshifts to correlation functions which we will project to angular
	# space
	thetachis, dipomp_int = mcfit.Hankel(k_grid, lowring=True)(pk_z, axis=1, extrap=True)

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
	differentials = dz_d_chi * dndz[1] ** 2

	# integrate over redshift kernel to get w(theta) a la Dipompeo+2017
	return 1 / (2 * np.pi) * np.trapz(differentials * np.transpose(interped_dipomp), x=dndz[0], axis=1)


def pk_z_to_cl_gg(pk_z, dndz, ells, k_grid, chi_zfunc, H_zfunc):
	chi_z, H_z = chi_zfunc(dndz[0]), H_zfunc(dndz[0])

	ks = np.outer(1. / chi_z_func(dndz[0]), (ells + 1 / 2.))

	dz_d_chi = (H_z / const.c.to(u.km / u.s).value)
	# product of redshift distributions, and dz/dchi
	differentials = dz_d_chi * dndz[1] / chi_z ** 2

	interped_power = []
	for j in range(len(dndz[0])):
		pk = pk_z[j]
		interped_power.append(10 ** interp1d(np.log10(k_grid), np.log10(pk))(np.log10(ks[j])))

	return np.trapz(differentials * np.transpose(interped_power), x=dndz[0], axis=1)

# Fourier transform input power spectra to correlation functions, convert to projected CFs wp(rp) if desired,
# and project to observable for given redshift distribution(s)
def pk_z_to_xi_r(pk_z, dndz, radii, k_grid, projected=True):

	if projected:
		#xis = np.array(abel.direct.direct_transform(xis, r=rgrid, direction='forward', backend='python'))
		rgrid, xis = mcfit.Hankel(k_grid, lowring=True)(pk_z, axis=1, extrap=True)
		xis /= (2 * np.pi)
	else:
		rgrid, xis = mcfit.P2xi(k_grid, lowring=True)(pk_z, axis=1, extrap=True)


	# trick to make interpolation work for logarithmically varying xi (propto r^-2)
	# multiply xi by r to make smooth in linear space, then divide r back out at end
	interpedxis = interp1d(rgrid, xis * rgrid)(radii) / radii

	return np.trapz(dndz[1] * np.transpose(interpedxis), x=dndz[0], axis=1)

def pk_z_to_xi_rp_pi(pk_z, dndz, beta_z, bz, rps, pis, k_grid, j3, j5):
	rp_grid, pi_grid = np.meshgrid(rps, pis)
	s_grid = np.sqrt(rp_grid ** 2 + pi_grid ** 2).ravel()
	mu_grid = rp_grid.ravel() / s_grid

	rgrid, xis = mcfit.P2xi(k_grid, lowring=True)(pk_z, axis=1, extrap=True)
	interpedxis = interp1d(rgrid, xis * rgrid)(s_grid) / s_grid
	beta_z = beta_z[:, np.newaxis]
	bz = bz[:, np.newaxis]


	xi0 = xi_monopole(interpedxis, beta_z)
	xi2 = xi_quadrupole(s_grid, interpedxis, beta_z, j3, bz)
	xi4 = xi_hexadecapole(s_grid, interpedxis, beta_z, j3, j5, bz)

	xirppi = legendre(0)(mu_grid) * xi0 + legendre(2)(mu_grid) * xi2 + legendre(4)(mu_grid) * xi4
	return np.trapz(dndz[1] * np.transpose(xirppi), x=dndz[0], axis=1).reshape(len(rps), len(pis)).ravel()





def pk_z_to_multipole(pk_z, dndz, beta_z, s, k_grid, j3, j5, bz, ell=0):
	rgrid, xis = mcfit.P2xi(k_grid, lowring=True)(pk_z, axis=1, extrap=True)
	interpedxis = interp1d(rgrid, xis * rgrid)(s) / s
	beta_z = beta_z[:, np.newaxis]
	bz = bz[:, np.newaxis]

	if ell == 0:
		xi_s = xi_monopole(interpedxis, beta_z)
	elif ell == 2:
		xi_s = xi_quadrupole(s, interpedxis, beta_z, j3, bz)
	elif ell == 4:
		xi_s = xi_hexadecapole(s, interpedxis, beta_z, j3, j5, bz)
	else:
		xi_s = None
	return np.trapz(dndz[1] * np.transpose(xi_s), x=dndz[0], axis=1)



# get cross spectrm C_ell between galaxy overdensity and gravitational lensing
def c_ell_kappa_g(pk_z, dndz, ells, k_grid, chi_z_func, H_z_func, lin_pk_z, lenskern):
	qsokern = H_z_func(dndz[0]) / const.c.to(u.km / u.s).value * dndz[1]

	integrand = (const.c.to(u.km / u.s).value * lenskern * qsokern / ((chi_z_func(dndz[0]) ** 2) * H_z_func(dndz[0])))
	ks = np.outer(1. / chi_z_func(dndz[0]), (ells + 1 / 2.))

	# P(k) is b^2 P_lin, so sqrt(P(k)) * sqrt(P_lin) = b * P_lin
	pk_z = np.sqrt(pk_z) * np.sqrt(lin_pk_z)

	ps_at_ks = []
	for j in range(len(dndz[0])):
		ps_at_ks.append(np.interp(ks[j], k_grid, pk_z[j]))

	ps_at_ks = np.array(ps_at_ks)

	integrand = integrand[:, None] * ps_at_ks

	return np.trapz(integrand, dndz[0], axis=0)


# defining the critical surface density for lensing
def sigma_crit(z):
	return ((const.c ** 2) / (4. * np.pi * const.G) *
			(apcosmo.angular_diameter_distance(paramobj.cmb_z) /
			((apcosmo.angular_diameter_distance(z) *
			apcosmo.angular_diameter_distance_z1z2(z, paramobj.cmb_z))))).decompose().to(
			u.solMass * cu.littleh / u.kpc ** 2, cu.with_H0(apcosmo.H0))


# model lensing convergence profile for stack
def cmb_kappa(pk_z, dndz, k_grid, lin_pk_z, l_beam=None, theta_grid=np.radians(np.linspace(0.001, 3, 1000))):
	import healpy as hp
	d_a = apcosmo.angular_diameter_distance(dndz[0]).to(u.kpc/cu.littleh, cu.with_H0(apcosmo.H0))	# kpc/h
	# P(k) is b^2 P_lin, so sqrt(P(k)) * sqrt(P_lin) = b * P_lin
	pk_z = np.sqrt(pk_z) * np.sqrt(lin_pk_z) * ((u.Mpc/cu.littleh) ** 3).to((u.kpc/cu.littleh) ** 3)

	# the average (matter) density of the universe
	rho_avg = col_cosmo.rho_m(dndz[0])*u.solMass*(cu.littleh**2)/(u.kpc**3)
	# product of redshift dependent variables outside the k-space integral
	a = ((rho_avg/(((1.+dndz[0])**3)*sigma_crit(dndz[0])*d_a**2)) / (2*np.pi)).value
	# outer product of k, (1+z) * D_A to get ells
	ls = np.outer(k_grid, (1+dndz[0]) *
				(apcosmo.angular_diameter_distance(dndz[0]).to(u.Mpc/cu.littleh, cu.with_H0(apcosmo.H0))).value)
	# outer product of ell(k,z) with theta
	ltheta = np.einsum("ij,k->ijk", ls, theta_grid)

	# Eq. 13 in Oguri and Hamana 2011
	integrand = np.transpose(j0(ltheta), axes=[2, 0, 1]) * a * ls * np.transpose(pk_z)
	# do integral over ell
	kappa_prof_z = []
	for j in range(len(dndz[0])):
		kappa_prof_z.append(np.trapz(integrand[:, :, j], x=ls[:, j]))
	# integral over dndz
	kappa_prof = np.trapz(np.transpose(kappa_prof_z) * dndz[1], x=dndz[0])
	if l_beam is not None:
		model_bl = hp.beam2bl(kappa_prof, theta_grid, lmax=4096)
		convolved = np.array(model_bl) * np.array(l_beam)
		return hp.bl2beam(convolved, theta_grid)
	return kappa_prof


# compute statistics for dark matter or tracers linearly biased against it
class halomodel(object):

	def __init__(self, dndz1, dndz2=None, littleh_units=True, transfer='eisenstein_hu'):

		self.dndz = redshift_helper.norm_z_dist(dndz1)
		self.zs = self.dndz[0]
		# if doing a cross correlationm, want the redshift distribution overlap of two samples
		if dndz2 is not None:
			dndz2 = redshift_helper.norm_z_dist(dndz2)
			self.dndz = (self.zs, np.sqrt(self.dndz[1] * dndz2[1]))

		#self.hm = hod_model.halomod_workspace(zs=self.zs, linpow=linpow)
		self.hm = ccl_tools.HOD_model(self.zs, transfer=transfer)
		self.lin_pk_z = self.hm.linpk_z
		self.pk_z = self.lin_pk_z
		self.k_grid = paramobj.k_space
		self.bias_relation = paramobj.bias_relation
		self.chizfunc = partial(chi_z_func, littleh_units=littleh_units)
		self.hzfunc = partial(Hubble_z, littleh_units=littleh_units)
		self.lens_kernel = lensing_kernel(lens_zs=self.zs, chi_z_func=chi_z_func, H0=self.hzfunc(0))
		self.bzs = np.ones_like(self.zs)

		rgrid, xis = mcfit.P2xi(self.k_grid, lowring=True)(self.lin_pk_z, axis=1, extrap=True)
		self.j3 = hamilton_j_interp(xis, rgrid, rgrid, 3)
		self.j5 = hamilton_j_interp(xis, rgrid, rgrid, 5)

	# reset the power spectrum according to an HOD if provided, or an effective mass-biased spectrum
	def set_powspec(self, hodparams=None, hodparams2=None, log_meff=None, log_meff_2=None,
					bias1=None, bias2=None, log_m_min1=None, log_m_min2=None,
					get1h=True, get2h=True):

		if hodparams is not None:
			self.pk_z = self.hm.hod_pk_a(hodparams=hodparams, get_1h=get1h, get_2h=get2h)
			if hodparams2 is not None:
				self.pk_z = np.sqrt(self.pk_z) * \
							np.sqrt(self.hm.hod_pk_a(hodparams=hodparams2, get_1h=get1h, get_2h=get2h))

		# if considering a population of halos with effective mass M rather than full HOD
		elif log_meff is not None:
			bz = self.bias_relation(M=10 ** log_meff, z=self.zs)
			bz2 = bz
			if log_meff_2 is not None:
				bz2 = self.bias_relation(M=10 ** log_meff_2, z=self.zs)
			self.pk_z = (bz * bz2)[:, None] * self.lin_pk_z
			self.bzs = np.sqrt(bz * bz2)
		# if halos with masses > M_min
		elif log_m_min1 is not None:
			bz = bias_tools.minmass_to_bias_z(log_minmass=log_m_min1, zs=self.zs)
			bz2 = bz
			if log_m_min2 is not None:
				bz2 = bias_tools.minmass_to_bias_z(log_minmass=log_m_min2, zs=self.zs)
			self.pk_z = (bz * bz2)[:, None] * self.lin_pk_z
			self.bzs = np.sqrt(bz * bz2)
		# or a simple constant bias with redshift
		elif bias1 is not None:
			bz, bz2 = bias1, bias1
			if bias2 is not None:
				bz2 = bias2
			self.pk_z = (bz * bz2) * self.lin_pk_z
			self.bzs = np.sqrt(bz * bz2) * np.ones_like(self.zs)
		else:
			self.pk_z = self.lin_pk_z
			self.bzs = np.ones_like(self.zs)



	def get_ang_cf(self, thetas):
		return pk_z_to_ang_cf(pk_z=self.pk_z, dndz=self.dndz, thetas=thetas, k_grid=self.k_grid,
							  chi_zfunc=self.chizfunc, H_zfunc=self.hzfunc)

	def get_c_ell_gg(self, ells):
		return pk_z_to_cl_gg(pk_z=self.pk_z, dndz=self.dndz, ells=ells, k_grid=self.k_grid, chi_zfunc=self.chizfunc,
							 H_zfunc=self.hzfunc)

	def get_binned_ang_cf(self, theta_bins, thetagrid=np.logspace(-3, 0, 100)):
		wtheta = self.get_ang_cf(thetas=thetagrid)
		return stats.binned_statistic(thetagrid, wtheta, statistic='mean', bins=theta_bins)[0]

	def get_spatial_cf(self, radii=np.logspace(-1., 1., 300), projected=True):
		return pk_z_to_xi_r(pk_z=self.pk_z, dndz=self.dndz, radii=radii, k_grid=self.k_grid, projected=projected)

	def get_binned_spatial_cf(self, radius_bins, sepgrid=np.logspace(-1., 1.6, 300), projected=True):
		wp = self.get_spatial_cf(radii=sepgrid, projected=projected)
		return stats.binned_statistic(sepgrid, wp, statistic='mean', bins=radius_bins)[0]

	def get_c_ell_kg(self, ells):
		return c_ell_kappa_g(pk_z=self.pk_z, dndz=self.dndz, ells=ells, k_grid=self.k_grid, chi_z_func=self.chizfunc,
							 H_z_func=self.hzfunc, lin_pk_z=self.lin_pk_z, lenskern=self.lens_kernel)

	def get_binned_c_ell_kg(self, ells, ell_bins=None, master_workspace=None):
		xpower = self.get_c_ell_kg(ells=ells)
		if master_workspace is not None:
			binned_xpow = master_workspace.decouple_cell(master_workspace.couple_cell([xpower]))[0]
		else:
			binned_xpow = 10 ** (stats.binned_statistic(ells, np.log10(xpower), statistic='median', bins=ell_bins)[0])
		return binned_xpow

	def get_binned_kappa_prof(self, theta_bins, l_beam=None, theta_grid=np.radians(np.linspace(0.001, 3, 1000))):
		kappa_theta = cmb_kappa(pk_z=self.pk_z, dndz=self.dndz, k_grid=self.k_grid, lin_pk_z=self.lin_pk_z,
								l_beam=l_beam, theta_grid=theta_grid)
		return stats.binned_statistic(np.degrees(theta_grid), kappa_theta, statistic='mean', bins=theta_bins)[0]

	def get_multipole(self, s, ell=0):
		return pk_z_to_multipole(pk_z=self.pk_z, dndz=self.dndz,
								 beta_z=beta_param(self.bzs, self.zs), bz=self.bzs,
								 s=s, k_grid=self.k_grid, j3=self.j3, j5=self.j5, ell=ell)

	def get_xi_rp_pi(self, rp, pi):
		return pk_z_to_xi_rp_pi(pk_z=self.pk_z, dndz=self.dndz, beta_z=beta_param(self.bzs, self.zs), bz=self.bzs,
								rps=rp, pis=pi, k_grid=self.k_grid, j3=self.j3, j5=self.j5)

	"""def get_multipole(self, s, ell=0):
		pkob = self.hm.biased_pk2dobj(self.bzs)
		xi_s_z = []
		for j in range(len(self.zs)):
			xi_s_z.append(self.hm.xi_s(z=self.zs[j], s=s, ell=ell, b=self.bzs[j], pk_a=pkob))
		return np.trapz(self.dndz[1] * np.transpose(xi_s_z), x=self.zs, axis=1)

	def get_xi_rp_pi(self, rp, pi):
		pkob = self.hm.biased_pk2dobj(self.bzs)
		xi_2d_z = []
		for j in range(len(self.zs)):
			xi_2d_z.append(self.hm.xi_rp_pi(z=self.zs[j], rps=rp, pis=pi, b=self.bzs[j], pk_a=pkob))
		return np.trapz(self.dndz[1] * np.transpose(xi_2d_z), x=self.zs, axis=1)"""

