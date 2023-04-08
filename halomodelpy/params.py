import numpy as np
from colossus.cosmology import cosmology

class param_obj(object):
	def __init__(self, kgrid_path=None, mgrid_path=None):
		# k space grid in units h/Mpc
		self.k_space = np.logspace(-3, 5, 1000)
		# mass grid in Msun/h
		self.mass_space = np.logspace(10, 15, 50)
		# set your cosmology in colossus
		self.col_cosmo = cosmology.setCosmology('planck18')
		# convert to astropy cosmology for some calculations
		self.apcosmo = self.col_cosmo.toAstropy()


		if kgrid_path is not None:
			self.k_space = np.load(kgrid_path, allow_pickle=True)
		if mgrid_path is not None:
			self.mass_space = np.load(mgrid_path, allow_pickle=True)

		# fix certain HOD parameters
		self.sigma_logM = 0.4
		self.M1_over_M0 = 12

		self.matter_power_source = 'camb'

		if self.matter_power_source == 'camb':
			import camb
			# Set up a new set of parameters for CAMB
			pars = camb.CAMBparams()
			# This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
			pars.set_cosmology(H0=self.apcosmo.H0.value, ombh2=self.apcosmo.Ob0*self.col_cosmo.h2,
								omch2=self.apcosmo.Odm0*self.col_cosmo.h2, omk=self.col_cosmo.Ok0)
			pars.InitPower.set_params(ns=self.col_cosmo.ns)
			#self.PK = camb.get_matter_power_interpolator(pars, zmax=5., nonlinear=False,
			#						hubble_units=True, k_hunit=True, kmax=np.max(np.log10(self.k_space)))
			self.cambpars = pars
		else:
			self.cambpars = None