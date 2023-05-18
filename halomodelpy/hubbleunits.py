from . import params
import astropy.units as u
from astropy.cosmology import units as cu
import numpy as np

paramobj = params.param_obj()
col_cosmo = paramobj.col_cosmo
apcosmo = paramobj.apcosmo

def add_h_to_mass(mass):
	return (mass * u.solMass).to(u.solMass / cu.littleh, cu.with_H0(apcosmo.H0)).value

def remove_h_from_mass(mass_per_h):
	return (mass_per_h * u.solMass / cu.littleh).to(u.solMass, cu.with_H0(apcosmo.H0)).value

def add_h_to_logmass(logmass):
	return np.log10(add_h_to_mass(10 ** logmass))

def remove_h_from_logmass(logmass):
	return np.log10(remove_h_from_mass(10 ** logmass))

def add_h_to_power(pk):
	return (pk * u.Mpc ** 3).to(((u.Mpc / cu.littleh) ** 3), cu.with_H0(apcosmo.H0)).value

def remove_h_from_power(pk):
	return (pk * ((u.Mpc / cu.littleh) ** 3)).to(u.Mpc ** 3, cu.with_H0(apcosmo.H0)).value

def add_h_to_wavenum(k):
	return (k / u.Mpc).to(cu.littleh / u.Mpc, cu.with_H0(apcosmo.H0)).value

def remove_h_from_wavenum(k):
	return (k * (cu.littleh / u.Mpc)).to(u.Mpc ** (-1), cu.with_H0(apcosmo.H0)).value

def add_h_to_scale(r):
	return (r * u.Mpc).to(u.Mpc / cu.littleh, cu.with_H0(apcosmo.H0)).value

def remove_h_from_scale(r):
	return (r * u.Mpc / cu.littleh).to(u.Mpc, cu.with_H0(apcosmo.H0)).value