Tools to calculate observables like correlation functions and lensing cross-powers in the halo model

Make environment with numpy, scipy, astropy
pip to get colossus, camb, halomod, mcfit, pyabel

import hm_calcs

hmobj = hm_calcs.halomodel(zs=(numpy grid of redshifts))

angular_correlation_function = hm_calcs.get_ang_cf(dndz=(zs, normalized histogram), thetas=np.logspace(-2, 0, 10))