import numpy as np
from . import interpolate_helper
from scipy import interpolate as interp


# ensure the redshift distribution is properly normalized
def norm_z_dist(dndz):
	return dndz[0], dndz[1] / np.trapz(dndz[1], x=dndz[0])


def dndz_from_z_list(zs, nbins, zrange=None):
	if np.min(zs) <= 0.:
		print('Warning: redshifts z<=0 passed, cutting')
		zs = zs[np.where(zs > 0.)]
	dndz, zbins = np.histogram(zs, bins=nbins, density=True, range=zrange)
	zcenters = interpolate_helper.bin_centers(zbins, method='mean')
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