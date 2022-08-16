import numpy as np

def bin_centers(binedges, method):
	if method == 'mean':
		return (binedges[1:] + binedges[:-1]) / 2
	elif method == 'geo_mean':
		return np.sqrt(binedges[1:] * binedges[:-1])
	else:
		return None