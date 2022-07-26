import numpy as np
import pandas as pd
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u


# convert ras and decs to galactic l, b coordinates
def equatorial_to_galactic(ra, dec):
	ra_decs = SkyCoord(ra, dec, unit='deg', frame='icrs')
	ls = np.array(ra_decs.galactic.l.radian * u.rad.to('deg'))
	bs = np.array(ra_decs.galactic.b.radian * u.rad.to('deg'))
	return ls, bs

# Equations 10-12 of F. Bianchini et al 2015 gives predicted S/N of lensing cross-correlation
def lensing_SN(coords, biased_hmobj, dndz, lens_map='Planck18', ell_limits=(100, 2000), ell_bins=None, ell_beam=None):
	# read lensing reconstruction noise file
	noisespec = pd.read_csv('/home/graysonpetter/ssd/Dartmouth/data/lensing_maps/%s/noise/nlkk.csv' % lens_map)
	ells, nl_plus_cl = noisespec['ell'], noisespec['nl_plus_cl']
	idxs = np.where((ells < ell_limits[1]) & (ells > ell_limits[0]))[0]
	ells, nl_plus_cl = ells[idxs], nl_plus_cl[idxs]

	# get prediction for cross spectrum given power spectrum
	theory_cl_kg = biased_hmobj.get_c_ell_kg(dndz, ells)
	# prediction for galaxy autospectrum
	gal_corr = biased_hmobj.get_c_ell_gg(dndz, ells)

	ls, bs = equatorial_to_galactic(coords[0], coords[1])
	samplesize = len(ls)
	lensmask = hp.read_map('/home/graysonpetter/ssd/Dartmouth/data/lensing_maps/%s/derived/mask.fits' % lens_map)
	pixinmask = np.where(lensmask[hp.ang2pix(nside=hp.npix2nside(len(lensmask)), theta=ls, phi=bs, lonlat=True)] == 1)
	ls, bs = ls[pixinmask], bs[pixinmask]
	print('%s percent of sample inside mask' % (len(ls)/samplesize))


	pix_of_sources = hp.ang2pix(nside=64, theta=ls, phi=bs, lonlat=True)
	npix = hp.nside2npix(nside=64)
	density_map = np.bincount(pix_of_sources, minlength=npix)

	fsky = len(density_map[np.where(density_map > 0)]) / npix
	dens_steradian = len(ls) / (fsky * 4 * np.pi)
	shotnoise = 1 / dens_steradian

	signoise = np.sqrt(((2 * ells + 1) * fsky * theory_cl_kg ** 2) /
					   ((theory_cl_kg ** 2) + nl_plus_cl * (gal_corr + shotnoise)))

	if ell_bins is not None:
		sn_bin = []
		for j in range(len(ell_bins) - 1):
			inbin = np.where((ells < ell_bins[j+1]) & (ells > ell_bins[j]))[0]

			sn_bin.append(np.sqrt(np.sum(np.square(signoise[inbin]))))
		return sn_bin
	elif ell_beam is not None:
		ell_beam = ell_beam[np.where((np.arange(len(ell_beam)) > ell_limits[0]) & (np.arange(len(ell_beam)) < ell_limits[1]))]
		return np.sqrt(np.sum(ell_beam*np.square(signoise)))
	return np.arange(ell_limits[0], ell_limits[1]-1), signoise