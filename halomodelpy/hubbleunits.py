from . import params
import astropy.units as u
from astropy.cosmology import units as cu

paramobj = params.param_obj()
col_cosmo = paramobj.col_cosmo
apcosmo = paramobj.apcosmo

def add_h_to_mass(mass):
	return (mass * u.solMass).to(u.solMass / cu.littleh, cu.with_H0(apcosmo.H0)).value

def remove_h_from_mass(mass_per_h):
	return (mass_per_h * u.solMass / cu.littleh).to(u.solMass, cu.with_H0(apcosmo.H0)).value