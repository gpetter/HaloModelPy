import numpy as np
from scipy import interpolate as interp

def bin_centers(binedges, method):
	if method == 'mean':
		return (binedges[1:] + binedges[:-1]) / 2
	elif method == 'geo_mean':
		return np.sqrt(binedges[1:] * binedges[:-1])
	else:
		return None

# ensure the redshift distribution is properly normalized
def norm_z_dist(dndz):
	return dndz[0], dndz[1] / np.trapz(dndz[1], x=dndz[0])


def dndz_from_z_list(zs, nbins, zrange=None):
	if np.min(zs) <= 0.:
		print('Warning: redshifts z<=0 passed, cutting')
		zs = zs[np.where(zs > 0.)]
	dndz, zbins = np.histogram(zs, bins=nbins, density=True, range=zrange)
	zcenters = bin_centers(zbins, method='mean')
	dndz = dndz / np.trapz(dndz, x=zcenters)
	return np.array(zcenters, dtype=np.float64), np.array(dndz, dtype=np.float64)

def fill_in_coarse_dndz(dndz, newzs):
	zs, dn_dz = dndz

	newdndz = np.interp(newzs, zs, dn_dz)
	return norm_z_dist((newzs, newdndz))

def spline_dndz(dndz, spline_k=4, smooth=0.05):

	zs = list(dndz[0])

	dn_dz = list(dndz[1])
	zs.insert(0, zs[0] - 0.01)
	dn_dz.insert(0, 0)
	zs.append(zs[-1] + 0.01)
	dn_dz.append(0)

	spl = interp.UnivariateSpline(zs, dn_dz, k=spline_k, s=smooth)
	return spl

	#return norm_z_dist((np.array(zcenters), np.array(spl(zcenters))))

def spl_interp_dndz(dndz, newzs, spline_k=4, smooth=0.05):
	spl = spline_dndz(dndz, spline_k=spline_k, smooth=smooth)
	return norm_z_dist((np.array(newzs), np.array(spl(newzs))))

def effective_z(dndz, dndz2=None):
	if dndz2 is not None:
		if not np.array_equal(dndz[0], dndz2[0]):
			print(dndz[0])
			print(dndz2[0])
			print('Redshift distribution grids do not match')
		dndz = (dndz[0], dndz[1]*dndz2[1])
	return np.average(dndz[0], weights=dndz[1])

def effective_lensing_z(dndz):
	from . import hm_calcs
	hmob = hm_calcs.halomodel(dndz)
	lenskern = hmob.lens_kernel
	chis = hmob.chi_z
	lensterm = lenskern / (chis ** 2)
	hz = hmob.Hz
	dchidz = hmob.dchidz
	integrand = hz * dndz[1] * lensterm * dchidz

	denmo = np.trapz(integrand, x=dndz[0])
	return np.trapz(integrand * dndz[0], x=dndz[0]) / denmo

	#lensterm /= np.trapz(lensterm, x=dndz[0])
	#lens_dist = dndz[1] * lensterm
	#return np.average(dndz[0], weights=lens_dist)
	#return np.trapz(lens_dist*dndz[0], x=dndz[0])


def dz2dvol(zbins):
	"""
	calculate comoving volume interval in dz, for converting between dn/dz and n(z), density per volume
	:param zbins:
	:return:
	"""
	from . import cosmo
	chis = cosmo.col_cosmo.comovingDistance(np.zeros(len(zbins)), zbins)
	outershellvols = 4 / 3 * np.pi * chis[1:] ** 3
	innershellvols = 4 / 3 * np.pi * chis[:-1] ** 3
	zcenters = bin_centers(zbins, method='mean')
	return zcenters, outershellvols - innershellvols
