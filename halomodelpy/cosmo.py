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

def rp2ell(rps, z, h_unit=True):
	"""
	Convert projected separations at redshift z to corresponding angular \ell modes
	"""
	theta = rp2angle(rps, z, h_unit)
	return np.sort(180. / theta)

def hmf_z(logM_hubble, z):
	return paramobj.hmf(10**logM_hubble, z)

def convert_halomass(mass_in, z, def_in, def_out):
	from colossus.halo import mass_adv
	mout, r_foo, c_foo = mass_adv.changeMassDefinitionCModel(mass_in, z, mdef_in=def_in,
															 mdef_out=def_out,
															 profile='nfw', c_model=paramobj.colossus_c_m)
	return mout
