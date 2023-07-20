import numpy as np
from . import params
from . import hubbleunits
paramobj = params.param_obj()
col_cosmo = paramobj.col_cosmo
apcosmo = paramobj.apcosmo


def chi(zs, h_unit=True):
	"""
	Comoving distance at redshift z
	:param zs:
	:param h_unit:
	:return:
	"""
	chis = apcosmo.comoving_distance(zs).value
	if h_unit:
		chis = hubbleunits.add_h_to_scale(chis)
	return chis

def rp2angle(rps, z, h_unit=True):
	"""
	Convert projected comoving separations at redshift z to corresponding angular scales in degrees
	:param rps:
	:param z:
	:param h_unit:
	:return:
	"""
	theta_scales = np.rad2deg(rps / chi(z, h_unit=h_unit))
	return theta_scales

def hmf_z(logM_hubble, z):
	return paramobj.hmf(10**logM_hubble, z)