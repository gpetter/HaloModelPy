Tools to calculate observables like correlation functions and lensing cross-powers in the halo model
As many steps as possible implemented with numpy for speed

Make environment with numpy, scipy, astropy
pip install colossus, camb, halomod, mcfit, pyabel

Set params in params.py

from halomodelpy import hm_calcs

hmobj = hm_calcs.halomodel(zs=(numpy grid of redshifts))

# get dark matter angular correlation function
angular_correlation_function = hm_calcs.get_ang_cf(dndz=(zs, normalized histogram), thetas=np.logspace(-2, 0, 100))

# or spatial (projected) CF w_p(r_p)
hm_calcs.get_spatial_cf(dndz, radii=np.logspace(-1., 1., 100))

# or CMB lensing cross power
hm_calcs.get_c_ell_kg(dndz, ls)


# set a different halo model, like one linearly biased w.r.t dark matter
hmobj.set_powspec(bias1=3)

# or an evolving bias with redshift using a fixed effective halo mass (little h units)
hmobj.set_powspec(log_meff=12.5)

# or an HOD model, with minimum halo mass and satellite power law \alpha for example
hmobj.set_powspec(hodparams=[12.5, 1.], modeltype='2param')

# now you can recalculate any observables again with the biased model