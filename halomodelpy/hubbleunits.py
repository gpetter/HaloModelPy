from . import params
import numpy as np
#import astropy.units as u
#from astropy.cosmology import units as cu

paramobj = params.param_obj()
col_cosmo = paramobj.col_cosmo
apcosmo = paramobj.apcosmo

littleh = col_cosmo.h

def add_h_to_mass(mass):
	return mass * littleh

def remove_h_from_mass(mass_per_h):
	return mass_per_h / littleh

def add_h_to_logmass(logmass):
	return logmass + np.log10(littleh)

def remove_h_from_logmass(logmass):
	return logmass - np.log10(littleh)

def add_h_to_power(pk):
	return pk * littleh ** 3

def remove_h_from_power(pk):
	return pk / (littleh ** 3)

def add_h_to_wavenum(k):
	return k / littleh

def remove_h_from_wavenum(k):
	return k * littleh

def add_h_to_scale(r):
	return r * littleh

def remove_h_from_scale(r):
	return r / littleh

def remove_h_from_density(dens):
	return dens * littleh ** 3

def add_h_to_density(dens):
	return dens / (littleh ** 3)