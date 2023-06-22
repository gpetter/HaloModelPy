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

def fill_in_coarse_dndz(dndz, newnbins):
	zs, dn_dz = dndz
	newzs = np.linspace(np.min(zs), np.max(zs), newnbins)
	newdndz = np.interp(newzs, zs, dn_dz)
	return newzs, newdndz

def spline_dndz(dndz, newzs, spline_k=3):
	zs = list(dndz[0])
	dn_dz = list(dndz[1])
	zs.insert(0, 0)
	dn_dz.insert(0, 0)
	zs.append(0)
	dn_dz.append(0)

	spl = interp.InterpolatedUnivariateSpline(zs, dn_dz, k=spline_k)

	return np.array(newzs), np.array(spl(newzs))

