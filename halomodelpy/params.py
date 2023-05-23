import numpy as np
from colossus.cosmology import cosmology
from functools import partial
from colossus.lss import bias
from colossus.lss import mass_function

class param_obj(object):
	def __init__(self):
		self.logk_min = -3
		self.logk_max = 2
		self.nks = 512

		self.cmb_z = 1090.

		# k space grid in units h/Mpc
		self.k_space = np.logspace(self.logk_min, self.logk_max, self.nks)
		# mass grid in Msun/h
		self.mass_space = np.logspace(10, 15, 50)
		# set your cosmology in colossus
		self.col_cosmo = cosmology.setCosmology('planck18')
		# convert to astropy cosmology for some calculations
		self.apcosmo = self.col_cosmo.toAstropy()
		self.mass_def = '200c'
		self.biasmodel = 'tinker10'
		self.hmfmodel = 'tinker08'
		self.c_m = 'Duffy08'

		self.bias_relation = partial(bias.haloBias, model=self.biasmodel, mdef=self.mass_def)
		self.hmf = partial(mass_function.massFunction, mdef=self.mass_def, model=self.hmfmodel, q_in='M', q_out='dndlnM')
