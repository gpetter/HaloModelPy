import numpy as np

class param_obj(object):
	def __init__(self, kgrid_path=None, mgrid_path=None):
		# k space grid in units h/Mpc
		self.k_space = np.logspace(-3, 5, 1000)
		# mass grid in Msun/h
		self.mass_space = np.logspace(10, 15, 50)

		if kgrid_path is not None:
			self.k_space = np.load(kgrid_path, allow_pickle=True)
		if mgrid_path is not None:
			self.mass_space = np.load(mgrid_path, allow_pickle=True)

		# fix certain HOD parameters
		self.sigma_logM = 0.4
		self.M1_over_M0 = 12

