import numpy as np
from colossus.cosmology import cosmology
from functools import partial
from colossus.lss import bias
from colossus.lss import mass_function

class param_obj(object):
	def __init__(self):
		# minimum log(k) for wavenumber integration grid
		self.logk_min = -5
		# maximum log(k) for grid
		self.logk_max = 3
		# number of k points on integration gride
		self.nks = 512

		# redshift of CMB
		self.cmb_z = 1090.

		# k space grid in units h/Mpc
		self.k_space = np.logspace(self.logk_min, self.logk_max, self.nks)
		# mass grid in Msun/h
		self.mass_space = np.logspace(10, 15, 50)
		# set your cosmology in colossus
		self.col_cosmo = cosmology.setCosmology('planck18')
		# convert to astropy cosmology for some calculations
		self.apcosmo = self.col_cosmo.toAstropy()
		# only 200x critical implemented
		self.mass_def = '200c'
		# bias/halo mass relation, choose between offerings from colossus
		self.biasmodel = 'tinker10'
		# halo mass function model, choose between offerings from colossus
		self.hmfmodel = 'tinker08'
		# concentration-mass relation, choose between offerings from colossus
		self.colossus_c_m = 'duffy08'
		# concentration-mass relation key to pass to ccl
		self.c_m = 'Duffy08'

		self.bias_relation = partial(bias.haloBias, model=self.biasmodel, mdef=self.mass_def)
		self.hmf = partial(mass_function.massFunction, mdef=self.mass_def, model=self.hmfmodel, q_in='M', q_out='dndlnM')
