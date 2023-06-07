import numpy as np
from . import interpolate_helper


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
	return (np.array(zcenters, dtype=np.float64), np.array(dndz, dtype=np.float64))