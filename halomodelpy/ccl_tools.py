import pyccl as ccl
import numpy as np
from . import hubbleunits
from . import params
paramobj = params.param_obj()
col_cosmo = paramobj.col_cosmo
apcosmo = paramobj.apcosmo


def z_to_a(zs):
	return 1. / (1. + zs)


def comoving_dist(zs, cosmo):
	a_arr = z_to_a(zs=zs)
	return cosmo.comoving_radial_distance(a=a_arr)

def hubble_parameter(zs, cosmo, littleh):
	a_arr = z_to_a(zs=zs)
	return cosmo.h_over_h0(a=a_arr) * littleh * 100.



class HOD_model(object):
	def __init__(self, z_space, transfer='eisenstein_hu'):
		self.littleh = col_cosmo.h
		self.cosmo = ccl.Cosmology(Omega_c=apcosmo.Odm0, Omega_b=apcosmo.Ob0,
							h=self.littleh, n_s=col_cosmo.ns, sigma8=col_cosmo.sigma8,
							transfer_function=transfer, matter_power_spectrum='linear')
		self.k_space = hubbleunits.remove_h_from_wavenum(paramobj.k_space)

		#self.cosmo.cosmo.spline_params.ELL_MAX_CORR = 1000000
		#self.cosmo.cosmo.spline_params.N_ELL_CORR = 5000

		self.mdef = ccl.halos.massdef.MassDef200c(c_m='Duffy08')
		self.c_m_relation = ccl.halos.concentration.ConcentrationDuffy08(mdef=self.mdef)
		self.hmf_mod = ccl.halos.hmfunc.MassFuncTinker08(cosmo=self.cosmo, mass_def=self.mdef)
		self.bias_mod = ccl.halos.hbias.HaloBiasTinker10(cosmo=self.cosmo, mass_def=self.mdef)
		self.hmc = ccl.halos.halo_model.HMCalculator(cosmo=self.cosmo, massfunc=self.hmf_mod,
													 hbias=self.bias_mod, mass_def=self.mdef)

		# flip this around since a is inversely propto z, so a is increasing
		#self.z_space = np.flip(np.linspace(0.1, 4., 16))
		self.a_space = z_to_a(np.array(z_space))

		# need to figure out what to do with f_c parameter, central fraction
		self.two_pt_hod_profile = ccl.halos.profiles_2pt.Profile2ptHOD()
		self.linpk_z = self.linear_pk_a()



	# kill off unphysical 1 halo power at scales less than kcut
	# a is dummy scalefactor, so cutoff doesn't change with redshift
	def large_scale_1h_suppresion_func(self, a, kcut=1e-2):
		return kcut * np.ones(len(self.k_space))

	# smooth transition
	# a is dummy scalefactor, so cutoff doesn't change with redshift
	# z=0 value from arXiv:2009.01858 is alpha = 0.719
	def smooth_1h_2h_transition_func(self, a, smooth_alpha=0.719):
		return smooth_alpha * np.ones(len(self.k_space))

	def nfw_profile(self):
		return ccl.halos.profiles.HaloProfileNFW(c_M_relation=self.c_m_relation)

	# set up density profile with HOD parameters
	def hod_profile(self, hodparams):
		if hodparams is None:
			return None
		mmin, sigm, m0, m1, alpha = hodparams
		mmin, m0, m1 = hubbleunits.remove_h_from_logmass([mmin, m0, m1])
		return ccl.halos.profiles.HaloProfileHOD(c_M_relation=self.c_m_relation, lMmin_0=mmin,
		                siglM_0=sigm, lM0_0=m0, lM1_0=m1, alpha_0=alpha)


	# calculate P(k, z) with given hod parameters
	def hod_pk_a(self, hodparams, hodparams2=None, get_1h=True, get_2h=True):
		return hubbleunits.add_h_to_power(np.array(ccl.halos.halo_model.halomod_power_spectrum(cosmo=self.cosmo,
								hmc=self.hmc, k=self.k_space,
								a=self.a_space, prof=self.hod_profile(hodparams), prof2=self.hod_profile(hodparams2),
								prof_2pt=self.two_pt_hod_profile,
								supress_1h=self.large_scale_1h_suppresion_func, normprof1=True, normprof2=True,
								get_1h=get_1h, get_2h=get_2h,
								smooth_transition=self.smooth_1h_2h_transition_func)))



	def linear_pk_a(self):
		pk_as = []
		for this_a in self.a_space:
			pk_as.append(ccl.power.linear_matter_power(cosmo=self.cosmo, k=self.k_space, a=this_a))
		return hubbleunits.add_h_to_power(np.array(pk_as))


	# ccl doesn't have built in tracer class for HOD
	# Just have a density tracer with no bias, then use the HOD power spectrum in the integral later
	def unbiased_density_tracer(self, dndz):
		return ccl.NumberCountsTracer(cosmo=self.cosmo, has_rsd=False, dndz=dndz, bias=(dndz[0], np.ones(len(dndz[0]))))

	def hod_pk_2d_obj(self, hodparams):
		return ccl.halos.halo_model.halomod_Pk2D(cosmo=self.cosmo, hmc=self.hmc, prof=self.hod_profile(hodparams),
		                        prof_2pt=self.two_pt_hod_profile, a_arr=self.a_space, lk_arr=np.log(self.k_space),
		                        normprof1=True, normprof2=True,
		                        supress_1h=self.large_scale_1h_suppresion_func,
		                        smooth_transition=self.smooth_1h_2h_transition_func)

	# do an angular autocorrelation of HOD halos
	# if thetas given, transform C_ell to W(theta)
	def hod_corr_gg(self, hodparams, dndz, thetas=None):
		pk_a_hod = self.hod_pk_2d_obj(hodparams, dndz[0])
		#hodtracer = self.hod_tracer(hodparams, dndz)
		gal_tracer = self.unbiased_density_tracer(dndz=dndz)
		ell_modes = 1+np.arange(500000)

		cl_gg = ccl.cls.angular_cl(cosmo=self.cosmo, cltracer1=gal_tracer, cltracer2=gal_tracer, ell=ell_modes,
		                           p_of_k_a=pk_a_hod)

		if thetas is not None:
			w_theta = ccl.correlations.correlation(cosmo=self.cosmo, ell=ell_modes, C_ell=cl_gg,
			                                       theta=thetas, type='NN', method='fftlog')
			return w_theta
		else:
			return cl_gg





	# cross correlation spectrum of galaxy density and CMB lensing
	def hod_c_ell_kg(self, hodparams, dndz):
		pk_a_hod = self.hod_pk_2d_obj(hodparams=hodparams, zs=dndz[0])
		hodtracer = self.unbiased_density_tracer(dndz=dndz)
		cmbtracer = ccl.tracers.CMBLensingTracer(cosmo=self.cosmo, z_source=1090.)
		ell_modes = 1 + np.arange(3000)

		c_ell_kg = ccl.cls.angular_cl(cosmo=self.cosmo, cltracer1=hodtracer, cltracer2=cmbtracer, ell=ell_modes,
		                           p_of_k_a=pk_a_hod)

		return c_ell_kg

	def hod_xi_of_r_z(self, hodparams, r_grid, z):
		hod_pk_a = self.hod_pk_2d_obj(hodparams=hodparams)
		a_return = z_to_a(z)
		return ccl.correlations.correlation_3d(cosmo=self.cosmo, a=a_return, r=r_grid, p_of_k_a=hod_pk_a)
